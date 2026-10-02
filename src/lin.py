from socp13 import *
import warnings; warnings.filterwarnings('ignore')
# LinDistFlow: P_ij = sum downstream injections (no losses); v_j = v_i - 2(r P + x Q)
nb=len(br); P=cp.Variable(nb); Q=cp.Variable(nb); v=cp.Variable(N); x=cp.Variable(N); zeq=cp.Variable()
Pd=cp.Variable(); Qd=cp.Variable(); ppv={k:cp.Variable() for k in PV}; qpv={k:cp.Variable() for k in PV}; pbs={k:cp.Variable() for k in BESS}
C=[0.95**2<=v,v<=1.05**2,0<=x,x<=1,0<=Pd,Pd<=Pd_max,cp.abs(Qd)<=Qd_max]
for k in PV: C+=[0<=ppv[k],ppv[k]<=PV[k],cp.SOC(np.hypot(PV[k],PVq),cp.hstack([ppv[k],qpv[k]]))]
for k in BESS: C+=[0<=pbs[k],pbs[k]<=BESS[k]]
for l,(f,t,c,L,S) in enumerate(br):
    C+=[v[idx[t]]==v[idx[f]]-2*(R[l]*P[l]+X[l]*Q[l]), cp.SOC(S,cp.hstack([P[l],Q[l]]))]
for bn in buses:
    i=idx[bn]; Pinj=-x[i]*PL[bn]; Qinj=-x[i]*QL[bn]
    if bn=='650': Pinj+=Pd; Qinj+=Qd
    if bn in PV: Pinj+=ppv[bn]; Qinj+=qpv[bn]
    if bn in BESS: Pinj+=pbs[bn]
    out=sum(P[l] for l,(f,*_ ) in enumerate(br) if f==bn); inn=sum(P[l] for l,(f,t,*_) in enumerate(br) if t==bn)
    outq=sum(Q[l] for l,(f,*_ ) in enumerate(br) if f==bn); innq=sum(Q[l] for l,(f,t,*_) in enumerate(br) if t==bn)
    C+=[out-inn==Pinj,outq-innq==Qinj]
    if PL[bn]>0: C+=[zeq>=w[bn]*(1-x[i])]
    else: C+=[x[i]==1]
pr=cp.Problem(cp.Minimize(a_d*cp.square(Pd)+b_d*Pd+8000*zeq),C); pr.solve(solver='CLARABEL')
print(pr.status,'Pd',Pd.value,'Qd',Qd.value,'z',zeq.value,'served',sum(x.value[idx[k]]*PL[k] for k in buses))
print('V lin',{b:round(float(np.sqrt(v.value[idx[b]])),4) for b in buses})
# AC check: fix shedding x from LinDist, run SOCP feasibility (min diesel) and see if feasible
xs=x.value.copy()
