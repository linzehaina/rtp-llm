"""Exhaustive order test of the actual compiled production FP32 score_key."""
import json, time
from pathlib import Path
import torch
import m31_topk_review_probe as ext

start = time.monotonic()
count = 0
chunk = 1 << 24
anchors = {}
for sign, descending in ((0, False), (0x80000000, True)):
    previous = None
    for first in range(0, 0x7f800001, chunk):
        last = min(first + chunk, 0x7f800001)
        raw = torch.arange(first + sign, last + sign, device='cuda', dtype=torch.int64).to(torch.int32)
        keys = ext.probe_score_keys(raw.view(torch.float32))
        good = keys[1:] < keys[:-1] if descending else keys[1:] > keys[:-1]
        assert bool(good.all()), (hex(sign), hex(first), 'non-monotonic key')
        a, b = keys[0].item(), keys[-1].item()
        if previous is not None:
            assert a < previous if descending else a > previous
        previous = b
        count += last - first
        if first == 0:
            anchors[hex(sign)] = a
        if last == 0x7f800001:
            anchors[hex(sign + 0x7f800000)] = b
        if first % (chunk * 32) == 0:
            print(json.dumps({'tested':count,'seconds':time.monotonic()-start}), flush=True)
        del raw, keys, good
assert anchors['0x0'] == anchors['0x80000000']
assert anchors['0x80000000'] > anchors['0xff800000'] > 0
assert anchors['0x7f800000'] > anchors['0x0']
result = {'pass':True,'non_nan_patterns':count,'chunk':chunk,'anchors':anchors,'seconds':time.monotonic()-start,'scope':'all finite FP32 encodings, signed zeros, positive/negative infinities; NaN handled before key in production'}
Path(__file__).with_name('exhaustive_keys.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
