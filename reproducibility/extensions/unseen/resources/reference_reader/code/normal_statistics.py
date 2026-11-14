"""Classical covariance/shrinkage and normal-response ridge kernels.

Exact numerical operations; callers supply normal arrays explicitly.
No paths, discovery, model initialization, automatic fitting or network IO.
"""
import numpy as np
import torch

LAMBDAS=[.001,.01,.1,1.]

def covariance(raw):
 flat=raw.reshape(-1,4,384);n=len(flat)
 sums=torch.zeros(384,device='cuda',dtype=torch.float64)
 second=torch.zeros(384,384,device='cuda',dtype=torch.float64);within=torch.zeros_like(second);between=torch.zeros_like(second)
 for i in range(0,n,4096):
  x=torch.from_numpy(flat[i:i+4096]).cuda().double();m=x.mean(1);d=(x-m[:,None]).reshape(-1,384);allx=x.reshape(-1,384)
  sums+=allx.sum(0);second+=allx.T@allx;within+=d.T@d;between+=m.T@m
 mu=sums/(4*n);total=second/(4*n)-torch.outer(mu,mu);W=within/(4*n);B=between/n-torch.outer(mu,mu)
 error=float((total-W-B).abs().max());assert error<1e-11,error
 return W.cpu().numpy(),total.cpu().numpy(),error

def transform(cov,alpha):
 tau=float(np.trace(cov)/384)
 if not np.isfinite(cov).all() or tau<=0:raise ValueError('Degenerate/nonfinite normal read covariance; no silent fallback')
 values,vectors=np.linalg.eigh(cov)
 if values.min() < -1e-9*max(values.max(),tau):raise ValueError('Indefinite covariance beyond roundoff')
 values=np.maximum(values,0)
 scale=1/np.sqrt((1-alpha)*values/tau+alpha)
 matrix=(vectors*scale)@vectors.T
 matrix=((matrix+matrix.T)*.5).astype(np.float32)
 assert np.isfinite(matrix).all()
 return matrix,dict(tau=tau,shrinkage=alpha,min_eigenvalue=float(values.min()),max_eigenvalue=float(values.max()),
  matrix_condition=float(scale.max()/scale.min()),dimensions=384,invertible=True)

def canonical_covariance(x):
 assert x.ndim==2 and x.shape[1]==384 and x.dtype==np.float32
 sums=torch.zeros(384,device='cuda',dtype=torch.float64);second=torch.zeros(384,384,device='cuda',dtype=torch.float64)
 for i in range(0,len(x),16384):
  z=torch.from_numpy(x[i:i+16384]).cuda().double();sums+=z.sum(0);second+=z.T@z
 mean=sums/len(x);cov=second/len(x)-torch.outer(mean,mean)
 assert bool(torch.isfinite(cov).all());return cov.cpu().numpy()

def fit(x,y,lam):
 x=torch.from_numpy(x).cuda().double();y=torch.from_numpy(y).cuda().double()
 mu=x.mean(0);bias=(y-x).mean(0);xc=x-mu;dc=y-x-bias
 cov=xc.T@xc/len(x);cross=xc.T@dc/len(x);tau=torch.trace(cov)/384
 assert bool(torch.isfinite(cov).all()) and float(tau)>0
 A=torch.linalg.solve(cov+lam*tau*torch.eye(384,device='cuda',dtype=torch.float64),cross)
 assert bool(torch.isfinite(A).all())
 return mu.cpu().numpy(),bias.cpu().numpy(),A.cpu().numpy().astype(np.float32),float(tau)

def project(x,model):
 mu,bias,A,_=model
 return (x.astype(np.float64)+bias+(x.astype(np.float64)-mu)@A).astype(np.float32)
