import os,sys,unittest,json,time
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source/rtp_llm/models_py/triton_kernels/sparse_msa/test'))
os.environ['M31_TOPK_TEST_EXTENSION']='m31_topk_review_probe'
import test_minimax_prefill_topk as gpu
import test_prefill_warp_dispatch as cpu
start=time.monotonic()
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(gpu.MinimaxPrefillTopkTest),unittest.defaultTestLoader.loadTestsFromTestCase(cpu.PrefillWarpDispatchTest)])
r=unittest.TextTestRunner(verbosity=2).run(suite)
data={'pass':r.wasSuccessful(),'methods':r.testsRun,'gpu_positive_cases':gpu.MinimaxPrefillTopkTest.cases,'errors':[(str(t),e) for t,e in r.errors],'failures':[(str(t),e) for t,e in r.failures],'seconds':time.monotonic()-start}
(root/'original_regression.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data),flush=True)
sys.exit(0 if r.wasSuccessful() else 1)
