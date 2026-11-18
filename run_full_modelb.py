
import os,sys,json,csv,math,time,random,hashlib,copy,collections,subprocess,traceback
from pathlib import Path
R=Path(__file__).resolve().parent.parent
W=Path('/data/taoran/cme_modelb_kd_20261031')
S=Path('/data/taoran/cme_model_b')
D=Path('/data/taoran/cme_model_a/deploy_20261016_1545/outputs/dataset')
T=S/'deploy_20261020/outputs/formal_20261020'
assert R.resolve()==R and R.is_relative_to(W) and W.resolve()==W
sys.dont_write_bytecode=True
for k,v in {'TMPDIR':'tmp','TMP':'tmp','TEMP':'tmp','XDG_CACHE_HOME':'cache','TORCH_HOME':'cache/torch','MPLCONFIGDIR':'config/mpl','XDG_CONFIG_HOME':'config'}.items():os.environ[k]=str(R/v)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from model_b import ModelB
from model_loader import ModelA
from dataset_b import CMEDatasetB,scaler_for_train_indices,FEATURES
from quality_filter import assert_quality_compliant
def dump(p,o):
 tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def save(p,o):
 tmp=p.with_suffix('.tmp');torch.save(o,tmp);tmp.replace(p)
def check(p):
 q=Path(p).resolve(strict=True)
 assert any(q.is_relative_to(r) for r in (S,W,D)),str(q)
 assert q.is_file() and q.stat().st_nlink==1,str(q)
 return q
def sha(p):
 h=hashlib.sha256()
 with check(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def csvout(p,rows):
 if not rows:return
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def seed():
 random.seed(395);np.random.seed(395);torch.manual_seed(395);torch.cuda.manual_seed_all(395)
def main():
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 device=torch.device('cuda:0') # launcher pins physical GPU by UUID
 status={'status':'PREPARING','pid':os.getpid()};dump(R/'status.json',status)
 inputs=[D/'image_labels.csv',D/'preprocessing.json',T/'physical_148_model_b.csv',T/'splits.json',T/'scaler_01.json',T/'fold_01.pt',T/'config.json']
 inputs+=list((R/'code').glob('*.py'))+[R/'code/quality_exclusions.json']
 hashes={str(check(p)):sha(p) for p in inputs}
 rows=list(csv.DictReader(check(D/'image_labels.csv').open(encoding='utf-8-sig')))
 sp=json.loads(check(T/'splits.json').read_text());ids={'train':sp['folds'][0]['train'],'validation':sp['folds'][0]['validation']}
 assert not set(ids['train'])&set(ids['validation']) and len(set(ids['train']+ids['validation']+sp['test']))==len(rows)
 assert_quality_compliant(rows,json.loads(check(D/'preprocessing.json').read_text()))
 # Check all manifest paths before any dataset may open them; read train/validation image content only.
 for row in rows:
  p=Path(row['processed_image_path'])
  check(p if p.is_absolute() else D/p)
 for i in ids['train']+ids['validation']:
  p=Path(rows[i]['processed_image_path']);p=check(p if p.is_absolute() else D/p);hashes[str(p)]=sha(p)
 dump(R/'input_hashes.json',hashes)
 sc=json.loads(check(T/'scaler_01.json').read_text())
 scaler=scaler_for_train_indices(D/'image_labels.csv',T/'physical_148_model_b.csv',ids['train'])
 assert tuple(sc['features'])==FEATURES and set(sc['training_event_ids'])==set(scaler.training_event_ids)
 assert np.allclose(sc['mean'],scaler.mean,rtol=1e-7,atol=1e-9) and np.allclose(sc['std'],scaler.scale,rtol=1e-7,atol=1e-9)
 scaler.mean=np.array(sc['mean']);scaler.scale=np.array(sc['std'])
 data={}
 for split in ids:
  ds=CMEDatasetB(D/'image_labels.csv',T/'physical_148_model_b.csv',indices=ids[split],scaler=scaler)
  values=[ds[i] for i in range(len(ds))]
  data[split]=TensorDataset(*(torch.stack(x) for x in zip(*values)))
  assert all(torch.isfinite(t).all() for t in data[split].tensors)
  assert data[split].tensors[0].shape[1:]==(1,256,256)
  del values,ds
 ty=data['train'].tensors[2];vy=data['validation'].tensors[2].double().numpy()
 width=.5;upper=(math.ceil(float(ty.max())/24)+1)*24;K=int(upper/width)
 edges=np.arange(K+1,dtype=np.float64)*width;centers=(edges[1:]+edges[:-1])/2
 assert float(ty.min())>=0 and float(ty.max())<upper
 hist=torch.bincount((ty/width).long(),minlength=K)
 def loader(split,shuffle=False):
  return DataLoader(data[split],batch_size=32,shuffle=shuffle,num_workers=0,generator=torch.Generator().manual_seed(395))
 def build():
  m=ModelB();m.fusion[4]=nn.Linear(32,K);ModelA.initialize(m.fusion[4]);return m
 seed();model=build();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 save(R/'initial_model.pt',initial);params=sum(v.numel() for v in model.parameters());del model
 protocol={'model':'Full ModelB: ModelA scalar +11 physics ->12->16->32->K','all_parameters_trainable':True,
 'parameters':params,'seed':395,'fold':1,'batch_size':32,'epochs_each':500,'smoke_epochs_each':1,
 'optimizer':'Adam','initial_lr':.008,'scheduler':'StepLR step_size10 gamma0.9',
 'hard':'one-hot bin cross entropy','soft':'Gaussian N(y,0.5h^2) CDF bin masses, truncated and normalized','soft_sigma_h':.5,
 'bin_width_h':width,'bins':K,'support_h':[0,upper],'support_rule':'ceil(train_max/24)*24+24; train-only',
 'train_range_h':[float(ty.min()),float(ty.max())],'empty_train_bins':int((hist==0).sum()),
 'validation_out_of_support':int(((vy<0)|(vy>=upper)).sum()),'selection':'minimum raw validation MAE of distribution mean',
 'primary_decode':'probability mean','secondary_decodes':['piecewise-uniform median','modal-bin center'],
 'bn':'best raw checkpoint; paired train-only cumulative BN recalibration','test_evaluated':False,
 'initialization':'same fresh Xavier state, no teacher/student pretrained weights','paper_scope':'CME histogram adaptation, not PROSWIN lognormal replication',
 'physical_gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'source':str(S),'readonly_data_root':str(D)}
 dump(R/'protocol.json',protocol)
 dump(R/'environment.json',{'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(device)})
 csvout(R/'bin_counts.csv',[{'bin':i,'left_h':float(edges[i]),'right_h':float(edges[i+1]),'train_count':int(hist[i])} for i in range(K)])
 edge_t=torch.tensor(edges,device=device)
 def soft(y):
  z=(edge_t[None,:]-y.double()[:,None])/(.5*math.sqrt(2))
  cdf=.5*(1+torch.erf(z));q=(cdf[:,1:]-cdf[:,:-1]).clamp_min(0)
  return (q/q.sum(1,keepdim=True)).float()
 def predict(m):
  m.eval();pp=[]
  with torch.no_grad():
   for x,p,y in loader('validation'):pp.append(m(x.to(device),p.to(device)).softmax(1).cpu().double().numpy())
  return np.concatenate(pp)
 def metrics(pr,y):
  pr=pr/pr.sum(1,keepdims=True);cum=pr.cumsum(1);n=len(y)
  def quantile(q):
   ix=(cum<q).sum(1).clip(max=K-1);before=np.where(ix>0,cum[np.arange(n),np.maximum(ix-1,0)],0)
   return ix*width+width*np.clip((q-before)/pr[np.arange(n),ix].clip(min=1e-300),0,1)
  mean=pr@centers;median=quantile(.5);mode=centers[pr.argmax(1)]
  delta=y[:,None]-centers
  eabs=np.where(abs(delta)>=width/2,abs(delta),(delta**2+(width/2)**2)/width)
  prevp=cum-pr;prevpc=np.cumsum(pr*centers,axis=1)-pr*centers
  crps=(pr*eabs).sum(1)-(pr*(centers*prevp-prevpc)).sum(1)-(pr**2).sum(1)*width/6
  m={'mae_h':float(abs(mean-y).mean()),'rmse_h':float(np.sqrt(((mean-y)**2).mean())),
   'median_mae_h':float(abs(median-y).mean()),'mode_mae_h':float(abs(mode-y).mean()),'crps_h':float(crps.mean())}
  predictions=[{'row_index':ids['validation'][j],'event_id':rows[ids['validation'][j]]['event_id'],'actual_h':float(y[j]),'predicted_h':float(mean[j]),'median_h':float(median[j]),'mode_h':float(mode[j]),'crps_h':float(crps[j])} for j in range(n)]
  ev=collections.defaultdict(list)
  for p in predictions:ev[p['event_id']].append(abs(p['actual_h']-p['predicted_h']))
  m['event_mae_h']=float(np.mean([np.mean(v) for v in ev.values()]))
  for cov in (.5,.8,.9):
   lo=quantile((1-cov)/2);hi=quantile((1+cov)/2);tag=str(int(cov*100))
   m['coverage_'+tag]=float(((y>=lo)&(y<=hi)).mean());m['interval_width_'+tag+'_h']=float((hi-lo).mean())
   for j,p in enumerate(predictions):p['lower_'+tag]=float(lo[j]);p['upper_'+tag]=float(hi[j])
  inside=(y>=0)&(y<upper);ix=np.floor(y/width).astype(int).clip(0,K-1)
  m['out_of_support']=int((~inside).sum())
  m['hard_bin_nll_inside']=float(-np.log(pr[np.arange(n)[inside],ix[inside]].clip(min=1e-300)).mean()) if inside.any() else None
  return m,predictions
 # Numerical checks: bin boundaries, normalized soft targets, CE equivalence and analytic uniform CRPS.
 assert torch.equal((torch.tensor([0.,.499,.5,1.])/width).long(),torch.tensor([0,0,1,2]))
 q=soft(torch.tensor([0.,.25,.5,upper-.01],device=device));assert torch.isfinite(q).all() and torch.allclose(q.sum(1),torch.ones(4,device=device))
 lg=torch.randn(4,K,device=device,requires_grad=True);ix=torch.tensor([0,1,2,3],device=device)
 assert torch.allclose(F.cross_entropy(lg,ix),-(F.one_hot(ix,K)*lg.log_softmax(1)).sum(1).mean())
 (-(q*lg.log_softmax(1)).sum(1).mean()).backward();assert torch.isfinite(lg.grad).all()
 pr=np.zeros((len(vy),K));pr[:,0]=1
 assert abs(metrics(pr,np.full(len(vy),width/2))[0]['crps_h']-width/12)<1e-10
 dump(R/'unit_checks.json',{'bin_boundaries':True,'soft_normalization':True,'ce_equivalence':True,'soft_gradients':True,'uniform_crps':True})
 results=[]
 for phase,epochs in [('smoke',1),('formal',500)]:
  for method in ['hard','soft']:
   md=R/phase/method;md.mkdir(parents=True,exist_ok=True)
   seed();m=build().to(device);m.load_state_dict(initial,strict=True)
   assert all(p.requires_grad for p in m.parameters())
   opt=torch.optim.Adam(m.parameters(),lr=.008);sched=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=.9)
   dl=loader('train',True);history=[];best=float('inf');start_epoch=1
   # True resume from last completed epoch if this process is explicitly relaunched.
   last=md/'last.pt'
   if last.exists():
    c=torch.load(last,map_location='cpu',weights_only=True);m.load_state_dict(c['model'],strict=True);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler'])
    history=c['history'];best=c['best'];start_epoch=c['epoch']+1
    torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);dl.generator.set_state(c['loader_rng']);random.setstate(c['python_rng'])
    ns=c['numpy_rng'];np.random.set_state((ns['algorithm'],np.array(ns['keys'],dtype=np.uint32),ns['pos'],ns['has_gauss'],ns['cached']))
   torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
   for epoch in range(start_epoch,epochs+1):
    t0=time.perf_counter();lr=opt.param_groups[0]['lr'];m.train();total=absum=0.;n=0;grad_image=grad_fusion=False
    for x,p,y in dl:
     x,p,y=x.to(device),p.to(device),y.to(device);opt.zero_grad(set_to_none=True);logits=m(x,p)
     assert logits.shape==(len(y),K)
     loss=F.cross_entropy(logits,(y/width).long()) if method=='hard' else -(soft(y)*logits.log_softmax(1)).sum(1).mean()
     assert torch.isfinite(loss);loss.backward()
     if epoch==1:
      grad_image |= any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in m.model_a.parameters())
      grad_fusion |= any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in m.fusion.parameters())
     opt.step();n+=len(y);total+=float(loss.detach())*len(y)
     absum+=float((logits.detach().softmax(1)@torch.tensor(centers,device=device,dtype=torch.float32)-y).abs().sum())
    if epoch==1:assert grad_image and grad_fusion,'Missing full ModelB gradients'
    pr=predict(m);val=float(abs(pr@centers-vy).mean())
    assert math.isfinite(val)
    improved=val<best
    if improved:
     best=val;save(md/'best_raw.pt',{'model':m.state_dict(),'epoch':epoch,'validation_mae_h':val,'scaler':sc,'protocol':protocol})
    sched.step();torch.cuda.synchronize(device)
    history.append({'epoch':epoch,'lr':lr,'train_ce':total/n,'train_mean_mae_h':absum/n,'val_mean_mae_h':val,'val_mean_rmse_h':float(np.sqrt(((pr@centers-vy)**2).mean())),'seconds':time.perf_counter()-t0})
    csvout(md/'history.csv',history)
    with (md/'bn_stats.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'buffers':{k:v.detach().cpu().tolist() for k,v in m.state_dict().items() if any(s in k for s in ['running_mean','running_var','num_batches_tracked'])}})+'\n')
    if epoch%10==0 or epoch==epochs:
     ns=np.random.get_state()
     save(last,{'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':epoch,'best':best,'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':dl.generator.get_state(),'python_rng':random.getstate(),'numpy_rng':{'algorithm':ns[0],'keys':ns[1].tolist(),'pos':ns[2],'has_gauss':ns[3],'cached':ns[4]}})
    dump(R/'status.json',{'status':'RUNNING','phase':phase,'method':method,'epoch':epoch,'budget':epochs,'best_raw_validation_mae_h':best,'last_validation_mae_h':val,'epoch_seconds':history[-1]['seconds'],'pid':os.getpid()})
    print(phase,method,epoch,'CE',total/n,'val',val,'best',best,'seconds',history[-1]['seconds'],flush=True)
   saved=torch.load(md/'best_raw.pt',map_location=device,weights_only=True);m.load_state_dict(saved['model'],strict=True)
   rawpr=predict(m);assert abs(float(abs(rawpr@centers-vy).mean())-saved['validation_mae_h'])<1e-5
   for bn in ['raw','train_recalibrated']:
    if bn=='raw':pr=rawpr
    else:
     before={k:v.detach().clone() for k,v in m.named_parameters()};m.eval()
     bns=[b for b in m.modules() if isinstance(b,nn.BatchNorm2d)];mom=[b.momentum for b in bns]
     for b in bns:b.reset_running_stats();b.momentum=None;b.train()
     with torch.no_grad():
      for x,p,y in loader('train'):m(x.to(device),p.to(device))
     for b,v in zip(bns,mom):b.momentum=v
     m.eval();assert all(torch.equal(v,dict(m.named_parameters())[k]) for k,v in before.items())
     save(md/'best_calibrated.pt',{'model':m.state_dict(),'scaler':sc,'protocol':protocol,'epoch':saved['epoch']})
     m.load_state_dict(torch.load(md/'best_calibrated.pt',map_location=device,weights_only=True)['model'],strict=True);pr=predict(m)
    met,pred=metrics(pr,vy);dump(md/('metrics_'+bn+'.json'),met);csvout(md/('validation_'+bn+'.csv'),pred)
    np.savez_compressed(md/('distribution_'+bn+'.npz'),probabilities=pr,edges=edges,row_indices=np.array(ids['validation']))
    if phase=='formal':results.append({'method':method,'split':'validation','bn':bn,'fold':1,**met})
   m.eval();x,p,_=next(iter(loader('validation')));x,p=x.to(device),p.to(device)
   with torch.no_grad():
    for _ in range(10):m(x,p).softmax(1)
    torch.cuda.synchronize(device);bt=time.perf_counter()
    for _ in range(50):m(x,p).softmax(1)
    torch.cuda.synchronize(device);lat=(time.perf_counter()-bt)*1000/50
   dump(md/'costs.json',{'parameters':params,'checkpoint_bytes':(md/'best_raw.pt').stat().st_size,'training_epoch_seconds':sum(h['seconds'] for h in history),'peak_cuda_bytes':torch.cuda.max_memory_allocated(device),'best_epoch':saved['epoch'],'latency_ms_batch32_including_softmax':lat,'warmup':10,'iterations':50})
   dump(md/'acceptance.json',{'status':'COMPLETED','epochs':epochs,'strict_reload':True,'calibration_parameters_unchanged':True,'all_modules_trainable':True})
   if phase=='formal':csvout(R/'results.csv',results)
   del m,opt,sched,saved;torch.cuda.empty_cache()
  if phase=='smoke':dump(R/'smoke/acceptance.json',{'status':'PASSED','both_full_modelb_methods':True,'full_backbone_gradients':True,'strict_reload_and_bn_checks':True})
 changed=[p for p,v in hashes.items() if sha(p)!=v];assert not changed,changed
 dump(R/'acceptance.json',{'status':'COMPLETED','model':'FULL_MODEL_B','epochs_each':500,'test_evaluated':False,'input_hashes_unchanged':True,'same_initialization':True})
 dump(R/'status.json',{'status':'COMPLETED','pid':os.getpid()})
 print('COMPLETE',json.dumps(results),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  dump(R/'status.json',{'status':'FAILED','exception':type(e).__name__,'message':str(e),'pid':os.getpid()})
  traceback.print_exc();raise
