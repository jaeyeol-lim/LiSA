"""Train graph-level LiSA on DrugOOD IC50 with the shared protocol."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
try:
    from .data import discover_data_root, load_splits
    from .model import JointGenerator, LiSAGIN
except ImportError:
    from data import discover_data_root, load_splits
    from model import JointGenerator, LiSAGIN

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain",choices=("assay","scaffold","size"),default="assay");p.add_argument("--subset",choices=("core","general","refined"),default="core");p.add_argument("--endpoint",choices=("ic50","ec50"),default="ic50")
    p.add_argument("--data-root",type=Path,default=discover_data_root());p.add_argument("--output-dir",type=Path);p.add_argument("--device",default="auto");p.add_argument("--seed",type=int,default=1)
    p.add_argument("--epochs",type=int,default=50);p.add_argument("--erm-pretrain-epochs",type=int,default=10);p.add_argument("--patience",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=128);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--lr",type=float,default=1e-3);p.add_argument("--weight-decay",type=float,default=0.)
    p.add_argument("--hidden-dim",type=int,default=128);p.add_argument("--num-layers",type=int,default=4);p.add_argument("--dropout",type=float,default=.1)
    p.add_argument("--loss-penalty-weight",type=float,default=.1,help="LiSA distribution-loss variance coefficient; sweep over 1,.1,.01,.001")
    p.add_argument("--kld-weight",type=float,default=.1);p.add_argument("--num-generators",type=int,default=3);p.add_argument("--inner-loop",type=int,default=20)
    p.add_argument("--selection-metric",choices=("accuracy","roc_auc"),default="accuracy");p.add_argument("--log-every",type=int,default=1)
    a=p.parse_args()
    if a.num_generators<2:p.error("--num-generators must be at least 2")
    return a
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def device_of(s):
    if s=="auto":return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d=torch.device(s)
    if d.type=="cuda" and not torch.cuda.is_available():raise RuntimeError(f"CUDA unavailable: {s}")
    return d
def loader(ds,a,shuffle):return DataLoader(ds,batch_size=a.batch_size,shuffle=shuffle,num_workers=a.num_workers,pin_memory=torch.cuda.is_available(),persistent_workers=a.num_workers>0)
def auc(y,s):
    if np.unique(y).size<2:return math.nan
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y,s))
@torch.no_grad()
def evaluate(model,dl,dev):
    model.eval();ys=[];scores=[];preds=[];loss=n=0
    for b in dl:
        b=b.to(dev);y=b.y.view(-1).long();z=model(b);loss+=float(F.cross_entropy(z,y,reduction="sum"));n+=y.numel();ys.append(y.cpu());scores.append(z.softmax(-1)[:,1].cpu());preds.append(z.argmax(-1).cpu())
    y=torch.cat(ys).numpy();s=torch.cat(scores).numpy();pr=torch.cat(preds).numpy()
    return {"loss":loss/max(n,1),"accuracy":float((pr==y).mean()),"roc_auc":auc(y,s),"count":n}
def erm_epoch(model,dl,dev,opt):
    model.train();total=n=0
    for b in dl:
        b=b.to(dev);y=b.y.view(-1).long();opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(b),y);loss.backward();opt.step();c=y.numel();total+=float(loss.detach())*c;n+=c
    return total/n
def lisa_parts(model,generators,b):
    emb=model.encode(b);base=F.cross_entropy(model.classify(b,emb),b.y.view(-1).long());losses=[base];klds=[]
    for gen in generators:
        kld,node_mask,edge_mask=gen(emb,b.edge_index,True);local=model.classify(b,model.encode(b,edge_mask),node_mask);losses.append(F.cross_entropy(local,b.y.view(-1).long()));klds.append(kld)
    return torch.stack(losses),torch.stack(klds)
def lisa_epoch(model,generators,dl,dev,model_opt,gen_opt,a):
    model.train();generators.train();total=mean_t=var_t=kld_t=0.;n=0
    for b in dl:
        b=b.to(dev)
        for _ in range(a.inner_loop):
            gen_opt.zero_grad(set_to_none=True);model_opt.zero_grad(set_to_none=True);losses,klds=lisa_parts(model,generators,b)
            # Official LiSA maximizes diversity between local risks while minimizing their mean and KL variance.
            local_sqrt=torch.sqrt(losses[1:]+1e-12);gen_loss=losses.mean()+a.kld_weight*klds.var(unbiased=True)-a.loss_penalty_weight*local_sqrt.var(unbiased=True)
            gen_loss.backward();gen_opt.step()
        model_opt.zero_grad(set_to_none=True);gen_opt.zero_grad(set_to_none=True);losses,klds=lisa_parts(model,generators,b);risk_mean=losses.mean();risk_var=losses.var(unbiased=True);loss=risk_mean+a.loss_penalty_weight*risk_var;loss.backward();model_opt.step()
        c=b.y.numel();total+=float(loss.detach())*c;mean_t+=float(risk_mean.detach())*c;var_t+=float(risk_var.detach())*c;kld_t+=float(klds.mean().detach())*c;n+=c
    return {"loss":total/n,"risk_mean":mean_t/n,"risk_variance":var_t/n,"kld":kld_t/n}
def train(a):
    seed_all(a.seed);dev=device_of(a.device);stem,splits=load_splits(a.data_root,a.subset,a.domain,a.endpoint);tr=loader(splits["train"],a,True);ev={k:loader(v,a,False) for k,v in splits.items() if k!="train"};sample=splits["train"][0]
    model=LiSAGIN(sample.x.shape[-1],sample.edge_attr.shape[-1],a.hidden_dim,a.num_layers,a.dropout).to(dev);out=a.output_dir or Path(__file__).resolve().parent/"outputs"/f"lisa_{stem}_seed{a.seed}_{int(time.time())}";out.mkdir(parents=True,exist_ok=True);best=out/"best.pt";history=[]
    opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    for e in range(1,a.erm_pretrain_epochs+1):
        loss=erm_epoch(model,tr,dev,opt);v=evaluate(model,ev["ood_val"],dev);history.append({"phase":"erm_pretrain","epoch":e,"train_loss":loss,"ood_val":v})
    generators=torch.nn.ModuleList([JointGenerator(a.hidden_dim) for _ in range(a.num_generators)]).to(dev);model_opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.weight_decay);gen_opt=torch.optim.Adam(generators.parameters(),lr=a.lr,weight_decay=a.weight_decay);bv=-math.inf;be=stale=0
    for e in range(1,a.epochs+1):
        tm=lisa_epoch(model,generators,tr,dev,model_opt,gen_opt,a);v=evaluate(model,ev["ood_val"],dev);value=v[a.selection_metric];history.append({"phase":"main","epoch":e,"train":tm,"ood_val":v})
        if e%a.log_every==0:print(f"epoch={e:03d} loss={tm['loss']:.4f} var={tm['risk_variance']:.4f} val_acc={v['accuracy']:.4f} val_auc={v['roc_auc']:.4f}")
        if value>bv:bv,be,stale=value,e,0;torch.save({"model":model.state_dict(),"generators":generators.state_dict(),"args":vars(a),"epoch":e},best)
        else:
            stale+=1
            if a.patience>0 and stale>=a.patience:break
    model.load_state_dict(torch.load(best,map_location=dev,weights_only=False)["model"]);metrics={k:evaluate(model,d,dev) for k,d in ev.items()};summary={"method":"LiSA","dataset":stem,"seed":a.seed,"best_epoch":be,"best_ood_val":bv,"selection_metric":a.selection_metric,"metrics":metrics,"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (out/"history.json").write_text(json.dumps(history,indent=2));(out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":train(parse_args())
