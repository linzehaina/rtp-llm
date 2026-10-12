import os,sys,json,unittest,time
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source/rtp_llm/models_py/triton_kernels/sparse_msa/test'))
os.environ['M31_TOPK_TEST_EXTENSION']='m31_topk_review_probe'
import test_prefill_topk_adversarial as test
names=sys.argv[1:] or ['test_fp32_extremes_ulp_subnormal_nan_and_fp16_boundaries','test_extreme_bias_counts_and_partial_causal_rows','test_full_legacy_compatibility_for_unique_numeric_extremes','test_three_streams_flag_lifetime_and_producer_consumer_order','test_int32_page_limit_and_unused_pages','test_empty_huge_stride_validation_does_not_wrap','test_real_pointer_offsets_above_int32','test_graph_replay_updates_metadata_and_fallback_flags_together','test_scalar_narrowing_bounds_before_empty_return']
start=time.monotonic()
suite=unittest.TestSuite([test.PrefillTopkAdversarialTest(n) for n in names])
result=unittest.TextTestRunner(verbosity=2).run(suite)
data={'pass':result.wasSuccessful(),'test_methods':result.testsRun,'positive_cases':test.PrefillTopkAdversarialTest.cases,'failures':[(str(t),e) for t,e in result.failures],'errors':[(str(t),e) for t,e in result.errors],'skipped':[(str(t),e) for t,e in result.skipped],'legacy_findings':test.PrefillTopkAdversarialTest.legacy_findings,'seconds':time.monotonic()-start}
output=os.environ.get('M31_REVIEW_RESULT','review_results.json')
(root/output).write_text(json.dumps(data,indent=2))
print(json.dumps({k:v for k,v in data.items() if k!='legacy_findings'}),flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
