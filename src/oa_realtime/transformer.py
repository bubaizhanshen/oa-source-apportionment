"""Causal Transformer and four-hour spectral context."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LOOKBACK = 8
MAX_AGE_MIN = 240
PATIENCE = 50

def build(x,times,indices):
    sequences=np.zeros((len(indices),LOOKBACK,x.shape[1]),dtype=float)
    valid=np.zeros((len(indices),LOOKBACK),dtype=bool)
    age=np.zeros((len(indices),LOOKBACK),dtype=float)
    for pos,i in enumerate(indices):
        candidates=np.arange(max(0,int(i)-LOOKBACK+1),int(i)+1)
        ages=(times[i]-times[candidates]).astype('timedelta64[s]').astype(float)/60.
        take=(ages>=0)&(ages<=MAX_AGE_MIN)
        candidates=candidates[take];ages=ages[take]
        if not len(candidates) or candidates[-1]!=i:raise ValueError('Current sample missing')
        first=LOOKBACK-len(candidates)
        sequences[pos,first:]=x[candidates];valid[pos,first:]=True;age[pos,first:]=ages
        if np.any(times[candidates]>times[i]):raise ValueError('Future sample in context')
    return sequences,valid,age


class SourceTransformer(nn.Module):
    def __init__(self,n_features,n_outputs,length,config):
        super().__init__()
        d=config['d_model']
        self.input=nn.Linear(n_features+1,d)
        self.position=nn.Parameter(torch.zeros(1,length,d))
        nn.init.normal_(self.position,std=.02)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=4*d,
            dropout=config['dropout'],activation='gelu',batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,num_layers=config['layers'],enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,n_outputs))

    def forward(self,x,mask):
        embedded=self.input(x)+self.position
        value=self.encoder(embedded,src_key_padding_mask=~mask)
        return self.head(value[:,-1,:])


def make_scaler(seq,mask,y,ids):
    # Each observed historical spectrum contributes once via its current token.
    current=seq[ids,-1,:]
    xm=current.mean(axis=0); xs=current.std(axis=0);xs=np.where(xs>1e-8,xs,1.)
    ym=y[ids].mean(axis=0);ys=y[ids].std(axis=0);ys=np.where(ys>1e-8,ys,1.)
    return xm,xs,ym,ys


def transform(seq,mask,age,stats):
    xm,xs,_,_=stats
    values=(seq-xm[None,None,:])/xs[None,None,:]
    values[~mask]=0.
    return np.concatenate([values,age[:,:,None]/240.],axis=2).astype(np.float32)


def train_model(seq,mask,age,y,train_idx,val_idx,config,seed,epochs,early_stop):
    torch.manual_seed(seed);np.random.seed(seed)
    stats=make_scaler(seq,mask,y,train_idx)
    xx=transform(seq,mask,age,stats); yy=((y-stats[2])/stats[3]).astype(np.float32)
    dataset=TensorDataset(torch.from_numpy(xx[train_idx]),torch.from_numpy(mask[train_idx]),torch.from_numpy(yy[train_idx]))
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=64,shuffle=True,generator=gen,num_workers=0)
    model=SourceTransformer(seq.shape[2],y.shape[1],seq.shape[1],config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-4)
    xv=torch.from_numpy(xx[val_idx]) if len(val_idx) else None
    mv=torch.from_numpy(mask[val_idx]) if len(val_idx) else None
    yv=torch.from_numpy(yy[val_idx]) if len(val_idx) else None
    best_loss=float('inf');best_epoch=0;best_state=None;stale=0;trace=[]
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0.;seen=0
        for xb,mb,yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=torch.mean((model(xb,mb)-yb)**2)
            if not torch.isfinite(loss):raise ValueError('Nonfinite Transformer loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();loss_sum+=float(loss.detach())*len(xb);seen+=len(xb)
        validation=None
        if len(val_idx):
            model.eval()
            with torch.no_grad():validation=float(torch.mean((model(xv,mv)-yv)**2))
            if not np.isfinite(validation):raise ValueError('Nonfinite validation loss')
            if validation<best_loss-1e-5:
                best_loss=validation;best_epoch=epoch;stale=0
                best_state={name:t.detach().clone() for name,t in model.state_dict().items()}
            else:stale+=1
        trace.append({'epoch':epoch,'training_loss':loss_sum/seen,'validation_loss':validation})
        if early_stop and stale>=PATIENCE:break
    if len(val_idx):
        if best_state is None:raise ValueError('No Transformer validation checkpoint')
        model.load_state_dict(best_state)
    else:best_epoch=epochs
    return model,stats,best_epoch,best_loss,trace


def predict(model, stats, sequences, masks, ages):
    inputs = transform(sequences, masks, ages, stats)
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(inputs), torch.from_numpy(masks)).numpy().astype(float)
    return np.maximum(output * stats[3][None, :] + stats[2][None, :], 0.0)
