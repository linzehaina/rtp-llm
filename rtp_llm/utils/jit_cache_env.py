"""Cache-directory configuration shared by test launchers and the runtime."""

import functools
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping
from urllib.parse import urlparse

_AUTOMATIC_CACHE_ENVS = "_RTP_LLM_AUTOMATIC_JIT_CACHE_ENVS"


def _automatic_cache_envs(environ: Mapping[str, str]) -> dict[str, str]:
    try:
        values = json.loads(environ.get(_AUTOMATIC_CACHE_ENVS, "{}"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items() if isinstance(v, str) and v.strip()}


@dataclass(frozen=True)
class CacheEnvConfig:
    """Capture opt-outs before automatic paths are exported to libraries."""

    explicit_envs: frozenset[str]


def read_cache_env(
    env_names: Iterable[str], environ: Mapping[str, str] | None = None
) -> CacheEnvConfig:
    environ = os.environ if environ is None else environ
    automatic = _automatic_cache_envs(environ)
    return CacheEnvConfig(
        frozenset(
            name
            for name in env_names
            if (value := environ.get(name, "").strip()) and automatic.get(name) != value
        )
    )


def configure_cache_env(
    env_name: str,
    directory: str | Path,
    *,
    automatic: bool | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Write a library path and its provenance together, preserving other entries.

    By default, retain an inherited automatic path's provenance. Legacy
    launcher defaults otherwise remain opt-outs, just as an ordinary preset is.
    """
    environ = os.environ if environ is None else environ
    marked = _automatic_cache_envs(environ)
    if automatic is None:
        current = environ.get(env_name, "").strip()
        automatic = bool(current and marked.get(env_name) == current)
    value = str(directory)
    environ[env_name] = value
    if automatic:
        marked[env_name] = value
    else:
        marked.pop(env_name, None)
    if marked:
        environ[_AUTOMATIC_CACHE_ENVS] = json.dumps(marked, sort_keys=True)
    else:
        environ.pop(_AUTOMATIC_CACHE_ENVS, None)


def safe_local_path(path: str | Path) -> Path | None:
    path = Path(os.path.abspath(Path(path).expanduser()))
    try:
        if any(candidate.is_symlink() for candidate in (path, *path.parents)):
            return None
        path = path.resolve()
    except OSError:
        return None
    remote = os.environ.get("REMOTE_JIT_DIR", "").strip()
    if remote and not urlparse(remote).scheme:
        remote_path = Path(remote).expanduser().resolve()
        if path == remote_path or remote_path in path.parents:
            return None
    return path


def ensure_writable_directory(path: str | Path) -> Path | None:
    path = safe_local_path(path)
    if path is None:
        return None
    try:
        try:
            path.mkdir(mode=0o700, parents=True)
        except FileExistsError:
            pass
        else:
            path.chmod(0o700)
        if path.stat().st_uid != os.getuid():
            return None
        with tempfile.NamedTemporaryFile(prefix=".write_probe_", dir=path):
            pass
    except OSError:
        return None
    return path


def _warn_on_explicit_cache_path(path: str | Path) -> None:
    # Explicit caches can deliberately be symlinked or shared; leave their modes alone.
    try:
        resolved = Path(os.path.abspath(Path(path).expanduser()))
        if resolved.is_symlink() or any(c.is_symlink() for c in resolved.parents):
            logging.warning("[JIT] explicit cache path is a symlink: %s", resolved)
        elif resolved.is_dir():
            st = resolved.stat()
            if st.st_uid != os.getuid():
                logging.warning(
                    "[JIT] explicit cache path is owned by uid %s: %s",
                    st.st_uid,
                    resolved,
                )
            elif st.st_mode & 0o022:
                logging.warning(
                    "[JIT] explicit cache path is group/world writable: %s", resolved
                )
    except OSError:
        pass


# The extension must be followed by a non-path character. Without the lookahead,
# object outputs like `foo_binding.cuda.o` falsely match the `.cu` prefix of
# `cuda`, so every cached_ops dir looks stale and concurrent purges delete dirs
# that another test's ninja is actively compiling into.
_NINJA_ABS_SOURCE = re.compile(r"(/[^\s:'\"]+\.(?:cu|cuh|h|hpp|cpp|cc))(?![\w.])")

# Any reference into a bazel runfiles tree. These paths are ephemeral: they are
# cleaned as soon as the owning test target finishes, while the shared JIT
# cache (and the sibling tests using it) lives on. Covers -isystem include
# dirs and header deps that carry no source extension, which _NINJA_ABS_SOURCE
# cannot see.
_NINJA_EPHEMERAL_PATH = re.compile(
    r"(/[^\s:'\"]*(?:nativelink/work|\.runfiles)/[^\s:'\"]+)"
)


def purge_stale_ninja_cached_ops(root: str | Path) -> int:
    """Drop FlashInfer cached_ops dirs whose ninja still names vanished sources.

    NativeLink runfiles are ephemeral. A shared /tmp/rtp-llm/.jit_cache snapshot can
    keep a build.ninja that points at a previous test's runfiles path; the next
    compile then aborts instead of rebuilding from the current package.
    """
    root = Path(root)
    if not root.is_dir():
        return 0
    wiped = 0
    for dirpath, _dirnames, filenames in os.walk(
        root, followlinks=False, onerror=lambda _exc: None
    ):
        if "cached_ops" not in Path(dirpath).parts or "build.ninja" not in filenames:
            continue
        ninja = Path(dirpath) / "build.ninja"
        try:
            # Join ninja `$`-newline continuations so split paths match whole.
            text = ninja.read_text(errors="replace").replace("$\n", "")
        except OSError:
            continue
        stale = False
        for match in _NINJA_ABS_SOURCE.finditer(text):
            source = Path(match.group(1))
            if source.is_absolute() and not source.exists():
                stale = True
                break
        if not stale:
            for match in _NINJA_EPHEMERAL_PATH.finditer(text):
                if not Path(match.group(1)).exists():
                    stale = True
                    break
        if not stale:
            continue
        shutil.rmtree(ninja.parent, ignore_errors=True)
        wiped += 1
    return wiped


_NINJA_BUILD_FAILED = "Ninja build failed"
_NEEDED_BY_TARGET = re.compile(r"needed by '([^']+)'")
_NINJA_ENTERING_DIR = re.compile(r"ninja: Entering directory [`']([^'`]+)['`]")


def wipe_stale_ninja_cached_ops_from_error(error: str) -> int:
    """Drop cached_ops dirs whose ninja build failed on vanished sources.

    Concurrent smoke tests share one JIT cache scope. A sibling test's runfiles
    can be cleaned after this test started, so the startup purge above cannot
    see it; ninja only notices mid-run. The error names the affected op dirs
    either explicitly ("needed by '<op>/<file>.o'") or as the build directory
    ("ninja: Entering directory `<op>'"); removing that dir lets the op
    regenerate from the current package.
    """
    wiped = 0
    candidates = [
        Path(match.group(1)).parent for match in _NEEDED_BY_TARGET.finditer(error)
    ]
    candidates += [
        Path(match.group(1)) for match in _NINJA_ENTERING_DIR.finditer(error)
    ]
    for cached_dir in candidates:
        if cached_dir.is_dir() and cached_dir.parent.name == "cached_ops":
            shutil.rmtree(cached_dir, ignore_errors=True)
            wiped += 1
    return wiped


def _retry_stale_ninja_build(build):
    @functools.wraps(build)
    def wrapper(self, *args, **kwargs):
        try:
            return build(self, *args, **kwargs)
        except RuntimeError as error:
            message = str(error)
            if _NINJA_BUILD_FAILED not in message:
                raise
            wiped = wipe_stale_ninja_cached_ops_from_error(message)
            if not wiped:
                raise
            logging.warning(
                "flashinfer JIT ninja referenced vanished sources; "
                "wiped %s cached_ops dirs and retrying",
                wiped,
            )
            return build(self, *args, **kwargs)

    wrapper._rtp_stale_retry = True  # type: ignore[attr-defined]
    return wrapper


def _patch_flashinfer_jit_core(module) -> None:
    spec_cls = module.JitSpec
    if getattr(spec_cls.build, "_rtp_stale_retry", False):
        return
    spec_cls.build = _retry_stale_ninja_build(spec_cls.build)
    logging.info("patched flashinfer JitSpec.build with stale-ninja retry")


class _FlashInferJitRetryFinder:
    """Post-import hook patching JitSpec.build once flashinfer.jit.core loads.

    A direct import here would pin whichever flashinfer copy happens to be on
    sys.path first (runfiles vs. staged cache), so the patch is deferred until
    the owning process imports flashinfer itself.
    """

    def find_spec(self, name, path=None, target=None):
        if name != "flashinfer.jit.core":
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(name, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader

        class _PatchingLoader:
            def create_module(self, mod_spec):
                create = getattr(loader, "create_module", None)
                return create(mod_spec) if create is not None else None

            def exec_module(self, module):
                loader.exec_module(module)
                try:
                    _patch_flashinfer_jit_core(module)
                except Exception:
                    logging.warning(
                        "flashinfer JIT stale-ninja retry patch failed",
                        exc_info=True,
                    )

        spec.loader = _PatchingLoader()
        return spec


def patch_flashinfer_jit_retry() -> None:
    """Make flashinfer JIT builds recover from stale runfiles ninja files.

    Wraps JitSpec.build so a ninja failure on vanished sources wipes the named
    cached_ops dir and rebuilds once from the current package. flashinfer
    serializes the rebuild with its own FileLock, so concurrent ranks stay
    consistent.
    """
    module = sys.modules.get("flashinfer.jit.core")
    if module is not None:
        try:
            _patch_flashinfer_jit_core(module)
        except Exception:
            logging.warning(
                "flashinfer JIT stale-ninja retry patch failed", exc_info=True
            )
        return
    if any(isinstance(finder, _FlashInferJitRetryFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _FlashInferJitRetryFinder())


def local_jit_fallback(name: str) -> Path:
    return Path(tempfile.gettempdir()).resolve() / f"rtp-llm-{os.getuid()}" / name


def configure_writable_cache_env(
    env_name: str,
    default_path: Path,
    fallback_name: str,
    cache_subpath: tuple[str, ...] = (),
) -> None:
    def writable_base(path):
        base = safe_local_path(path)
        if base is None:
            return None
        cache_dir = base.joinpath(*cache_subpath) if cache_subpath else base
        return base if ensure_writable_directory(cache_dir) is not None else None

    requested = os.environ.get(env_name, "").strip()
    if requested:
        _warn_on_explicit_cache_path(requested)
        directory = Path(os.path.abspath(Path(requested).expanduser()))
        configure_cache_env(env_name, directory)
    else:
        directory = writable_base(default_path)
        if directory is None:
            directory = writable_base(local_jit_fallback(fallback_name))
        if directory is None:
            raise OSError(f"no writable {fallback_name} cache directory")
        configure_cache_env(env_name, directory, automatic=True)
