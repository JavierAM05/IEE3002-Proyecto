from socp13 import *
import warnings; warnings.filterwarnings('ignore')

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

xs=x.value.copy()


print("\n--- Ejecutando AC Feasibility Check para LinDistFlow ---")


cii_ac = cp.Variable(N); cij_ac = cp.Variable(nb); sij_ac = cp.Variable(nb)
Pd_ac = cp.Variable(); Qd_ac = cp.Variable()
ppv_ac = {k: cp.Variable() for k in PV}; qpv_ac = {k: cp.Variable() for k in PV}
pbs_ac = {k: cp.Variable() for k in BESS}

Pf_ac = []; Pt_ac = []; Qf_ac = []; Qt_ac = []
for l, (f, t, *_) in enumerate(br):
    i, j = idx[f], idx[t]
    Pf_ac.append(g[l]*(cii_ac[i]-cij_ac[l])-b[l]*sij_ac[l])
    Qf_ac.append(-b[l]*(cii_ac[i]-cij_ac[l])-g[l]*sij_ac[l])
    Pt_ac.append(g[l]*(cii_ac[j]-cij_ac[l])+b[l]*sij_ac[l])
    Qt_ac.append(-b[l]*(cii_ac[j]-cij_ac[l])+g[l]*sij_ac[l])

C_ac = [0.95**2 <= cii_ac, cii_ac <= 1.05**2, 0 <= Pd_ac, Pd_ac <= Pd_max, cp.abs(Qd_ac) <= Qd_max]

for l, (f, t, *_) in enumerate(br):
    i, j = idx[f], idx[t]
    C_ac += [cp.SOC(cii_ac[i] + cii_ac[j], cp.hstack([2*cij_ac[l], 2*sij_ac[l], cii_ac[i] - cii_ac[j]]))]
    Smax = br[l][4]
    C_ac += [cp.SOC(Smax, cp.hstack([Pf_ac[l], Qf_ac[l]])), cp.SOC(Smax, cp.hstack([Pt_ac[l], Qt_ac[l]]))]

for k in PV: C_ac += [0 <= ppv_ac[k], ppv_ac[k] <= PV[k], cp.SOC(np.hypot(PV[k], PVq), cp.hstack([ppv_ac[k], qpv_ac[k]]))]
for k in BESS: C_ac += [0 <= pbs_ac[k], pbs_ac[k] <= BESS[k]]

for bn in buses:
    i = idx[bn]
    Pinj = -xs[i] * PL[bn]
    Qinj = -xs[i] * QL[bn]
    if bn == '650': Pinj += Pd_ac; Qinj += Qd_ac
    if bn in PV: Pinj += ppv_ac[bn]; Qinj += qpv_ac[bn]
    if bn in BESS: Pinj += pbs_ac[bn]

    fp = sum(Pf_ac[l] for l, (f, t, *_) in enumerate(br) if f == bn) + sum(Pt_ac[l] for l, (f, t, *_) in enumerate(br) if t == bn)
    fq = sum(Qf_ac[l] for l, (f, t, *_) in enumerate(br) if f == bn) + sum(Qt_ac[l] for l, (f, t, *_) in enumerate(br) if t == bn)
    C_ac += [fp == Pinj, fq == Qinj]

prob_ac_check = cp.Problem(cp.Minimize(Pd_ac), C_ac)
prob_ac_check.solve(solver='CLARABEL')

print(f"Estado de factibilidad AC: {prob_ac_check.status}")
if prob_ac_check.status not in ["optimal", "optimal_inaccurate"]:
    print("Conclusión: El despacho de LinDistFlow es INFACTIBLE en la red AC real.")