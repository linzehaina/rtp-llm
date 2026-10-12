"""Each invalid device metadata case runs in a fresh process/context."""
import sys,json,torch
import m31_topk_review_probe as ext
name=sys.argv[1]
s=torch.ones((4,8,579),device='cuda')
ti=torch.empty((4,8,16),device='cuda',dtype=torch.int32)
cu=torch.tensor([0,3,8],device='cuda',dtype=torch.int32)
pre=torch.tensor([0,0],device='cuda',dtype=torch.int32)
qmax=5
if name=='negative_prefix': pre[0]=-1
elif name=='max_int_prefix': pre[0]=2147483647
elif name=='bad_cu_start': cu[0]=1
elif name=='bad_cu_end': cu[-1]=9
elif name=='descending_cu': cu[1]=7;cu[2]=6
elif name=='qmax_too_small': qmax=4
else:raise ValueError(name)
try:
 ext.minimax_prefill_topk(s,ti,ti,ti,cu,pre,qmax,16,0,0,False,579)
 torch.cuda.synchronize()
except RuntimeError as e:
 assert 'device-side assert' in str(e),str(e)
 print(json.dumps({'case':name,'pass':True,'error':'device-side assert triggered before accepting malformed metadata'}),flush=True)
 sys.exit(0)
raise AssertionError('Malformed metadata silently accepted: '+name)
