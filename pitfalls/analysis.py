"""Why long context "stabilizes": exact closed-loop spectrum of the learned history policy
(ell=1) in the lifted state z = [x_t..x_{t-r+1}, u_{t-1}..u_{t-r+1}], plus the open-loop baseline.
Run from repo root: python3 pitfalls/analysis.py"""
import sys; from pathlib import Path; R=Path(__file__).resolve().parent; sys.path[:0]=[str(R.parent), str(R)]
import numpy as np
from pitfalls import challenging_pair, train_set
from model import LinearSystem, fit_policy
mu=0.2; A1,_,K1,_=challenging_pair(mu)
def lifted(W, r, A, C):
    # z=[x_t..x_{t-r+1}, u_{t-1}..u_{t-r+1}], feature=[y_{t-r+1..t}, u_{t-r+1..t-1}]
    n=2; N=n*r+2*(r-1)
    p=C.shape[0]; F=np.zeros((p*r+2*(r-1),N))
    for k in range(r):  # y index k corresponds to x_{t-r+1+k} = z block r-1-k
        F[k*p:(k+1)*p, (r-1-k)*n:(r-k)*n]=C
    for k in range(r-1): # u_{t-r+1+k} = u block r-2-k
        F[p*r+2*k:p*r+2*k+2, n*r+2*(r-2-k):n*r+2*(r-1-k)]=np.eye(2)
    G=W[:2]@F  # u_t = G z
    M=np.zeros((N,N))
    M[:n,:n]=A; M[:n]+=G
    for k in range(1,r): M[k*n:(k+1)*n,(k-1)*n:k*n]=np.eye(n)
    if r>1:
        M[n*r:n*r+2]=G
        for k in range(1,r-1): M[n*r+2*k:n*r+2*k+2, n*r+2*(k-1):n*r+2*k]=np.eye(2)
    return M,G
for tag,C,exc in [('full narrow',np.eye(2),0),('partial narrow',np.array([[0.,1]]),0),('full excited',np.eye(2),.3)]:
    print(tag)
    sysm=LinearSystem(A1,np.eye(2),C); rng=np.random.default_rng(618)
    tr=train_set(sysm,K1,32,40,exc,rng)
    for r in [1,2,3,4,6,8]:
        pol,d=fit_policy(tr,r,16); M,G=lifted(pol.weights,r,A1,C)
        ev=np.linalg.eigvals(M); 
        # response of u_t to current x1 and to e1 one step ago (through x2 lag)
        print(f' r={r} rank={d["feature_rank"]}/{d["feature_dim"]} rho={max(abs(ev)):.4f} |W|={np.linalg.norm(pol.weights[:2]):.2f} gain_on_x1_t={G[:,0].round(3)}')
print('rho(A1)=',max(abs(np.linalg.eigvals(A1))))
# stationary RMS of open loop A1 with noise .01: solve Lyapunov
P=np.eye(2)*1e-4
for _ in range(5000): P=A1@P@A1.T+1e-4*np.eye(2)
print('open-loop stationary RMS',np.sqrt(np.trace(P)))
sysm=LinearSystem(A1,np.eye(2),np.array([[0.,1]])); tr=train_set(sysm,K1,32,40,0,np.random.default_rng(618))
pol,_=fit_policy(tr,8,16); print('r=8 partial: weight of u1_t on y lags (oldest->newest)',pol.weights[0,:8].round(4))
print('excited data, x1 in windows for t>=1:', max(abs(t['x'][1:,0]).max() for t in train_set(LinearSystem(A1,np.eye(2),np.eye(2)),K1,32,40,.3,np.random.default_rng(1))))
