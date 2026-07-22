"""LiSA joint masking generators with a DrugOOD GIN."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing,global_add_pool
class MaskGINConv(MessagePassing):
    def __init__(self,h,e):
        super().__init__(aggr="add");self.ee=nn.Linear(e,h);self.eps=nn.Parameter(torch.zeros(1));self.mlp=nn.Sequential(nn.Linear(h,2*h),nn.BatchNorm1d(2*h),nn.ReLU(),nn.Linear(2*h,h))
    def forward(self,x,ei,ea,mask=None):return self.mlp((1+self.eps)*x+self.propagate(ei,x=x,ea=self.ee(ea.float()),mask=mask))
    def message(self,x_j,ea,mask):
        m=F.relu(x_j+ea);return m if mask is None else m*mask
class LiSAGIN(nn.Module):
    def __init__(self,node_dim,edge_dim,h=128,layers=4,drop=.1):
        super().__init__();self.ne=nn.Linear(node_dim,h);self.cs=nn.ModuleList([MaskGINConv(h,edge_dim) for _ in range(layers)]);self.ns=nn.ModuleList([nn.BatchNorm1d(h) for _ in range(layers)]);self.fc=nn.Linear(h,2);self.drop=drop
    def encode(self,b,edge_mask=None):
        x=self.ne(b.x.float())
        for i,(c,n) in enumerate(zip(self.cs,self.ns)):
            x=n(c(x,b.edge_index,b.edge_attr,edge_mask));x=F.relu(x) if i+1<len(self.cs) else x;x=F.dropout(x,self.drop,training=self.training)
        return x
    def classify(self,b,node_emb,node_mask=None):
        x=node_emb if node_mask is None else node_emb*node_mask;return self.fc(global_add_pool(x,b.batch))
    def forward(self,b,node_mask=None,edge_mask=None):return self.classify(b,self.encode(b,edge_mask),node_mask)
class JointGenerator(nn.Module):
    def __init__(self,h):super().__init__();self.net=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(self,emb,edge_index,training=True):
        logits=self.net(emb).clamp(-10,10)
        if training:
            eps=torch.empty_like(logits).uniform_(1e-4,1-1e-4);node=((logits+eps.log()-(1-eps).log())).sigmoid()
        else:node=logits.sigmoid()
        row,col=edge_index;edge=.5*(node[row]+node[col]);kld=(node*torch.log(node/.5+1e-8)+(1-node)*torch.log((1-node)/.5+1e-8)).mean();return kld,node,edge
