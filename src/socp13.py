import numpy as np, cvxpy as cp
Sb=1.0; Vb=4.16; Zb=Vb**2/Sb; mi=5280.
z={'601':(0.3465,1.0179),'602':(0.7526,1.1814),'603':(1.3294,1.3471),'604':(1.3238,1.3569),
   '605':(1.3292,1.3475),'606':(0.7982,0.4463),'607':(1.3425,0.5124)}
br=[('650','632','601',2000,2.5),('632','633','602',500,1.0),('633','634','XF',0,0.5),
    ('632','645','603',500,1.0),('645','646','603',300,1.0),('632','671','601',2000,2.5),
    ('671','680','601',1000,1.0),('671','684','604',300,1.0),('684','611','605',300,1.0),
    ('684','652','607',800,1.0),('671','692','SW',0,2.5),('692','675','606',500,1.5)]
buses=['650','632','633','634','645','646','671','680','684','611','652','692','675']
idx={b:k for k,b in enumerate(buses)}; N=len(buses)
R=[];X=[]
for f,t,c,L,_ in br:
    if c=='XF': r,x=0.011*Sb/0.5,0.02*Sb/0.5
    elif c=='SW': r,x=1e-4,1e-4
    else: r,x=z[c][0]*L/mi/Zb, z[c][1]*L/mi/Zb
    R.append(r);X.append(x)
R=np.array(R);X=np.array(X); y=1/(R+1j*X); g=y.real; b=y.imag
# loads MW, Mvar (IEEE13 totals, 670 lumped into 671)
PL=dict.fromkeys(buses,0.);QL=dict.fromkeys(buses,0.)
for k,(p,q) in {'634':(.400,.290),'645':(.170,.125),'646':(.230,.132),'652':(.128,.086),
    '671':(1.155+.200,.660+.116),'675':(.843,.462),'692':(.170,.151),'611':(.170,.080)}.items():
    PL[k]=p;QL[k]=q
w=dict.fromkeys(buses,1.0); w['671']=3.0; w['652']=2.0   # 671: hospital/critical; 652: water pumping
PV={'675':0.45,'680':0.30}; PVq=0.20
BESS={'671':0.25}
Pd_max,Qd_max=2.0,1.5
a_d,b_d=20.,230.; cP,cQ=150.,30.; dt=1.0

def solve(c_eq, verbose=False):
    nb=len(br)
    cii=cp.Variable(N); cij=cp.Variable(nb); sij=cp.Variable(nb)
    x=cp.Variable(N); zeq=cp.Variable()
    Pd=cp.Variable(); Qd=cp.Variable()
    ppv={k:cp.Variable() for k in PV}; qpv={k:cp.Variable() for k in PV}
    pbs={k:cp.Variable() for k in BESS}
    Pf=[];Pt=[];Qf=[];Qt=[]
    for l,(f,t,*_) in enumerate(br):
        i,j=idx[f],idx[t]
        Pf.append(g[l]*(cii[i]-cij[l])-b[l]*sij[l]); Qf.append(-b[l]*(cii[i]-cij[l])-g[l]*sij[l])
        Pt.append(g[l]*(cii[j]-cij[l])+b[l]*sij[l]); Qt.append(-b[l]*(cii[j]-cij[l])+g[l]*sij[l])
    C=[0.95**2<=cii, cii<=1.05**2, 0<=x, x<=1, 0<=Pd, Pd<=Pd_max, cp.abs(Qd)<=Qd_max]
    for l,(f,t,*_) in enumerate(br):
        i,j=idx[f],idx[t]
        C+=[cp.SOC(cii[i]+cii[j], cp.hstack([2*cij[l],2*sij[l],cii[i]-cii[j]]))]
        Smax=br[l][4]; C+=[cp.SOC(Smax,cp.hstack([Pf[l],Qf[l]])), cp.SOC(Smax,cp.hstack([Pt[l],Qt[l]]))]
    for k in PV: C+=[0<=ppv[k],ppv[k]<=PV[k],cp.SOC(np.hypot(PV[k],PVq),cp.hstack([ppv[k],qpv[k]]))]
    for k in BESS: C+=[0<=pbs[k],pbs[k]<=BESS[k]]
    for bn in buses:
        i=idx[bn]
        Pinj=-x[i]*PL[bn]; Qinj=-x[i]*QL[bn]
        if bn=='650': Pinj+=Pd; Qinj+=Qd
        if bn in PV: Pinj+=ppv[bn]; Qinj+=qpv[bn]
        if bn in BESS: Pinj+=pbs[bn]
        fp=sum(Pf[l] for l,(f,t,*_) in enumerate(br) if f==bn)+sum(Pt[l] for l,(f,t,*_) in enumerate(br) if t==bn)
        fq=sum(Qf[l] for l,(f,t,*_) in enumerate(br) if f==bn)+sum(Qt[l] for l,(f,t,*_) in enumerate(br) if t==bn)
        C+=[fp==Pinj, fq==Qinj]
        if PL[bn]>0: C+=[zeq>=w[bn]*(1-x[i])*dt]
        else: C+=[x[i]==1]
    Ploss=sum(Pf[l]+Pt[l] for l in range(nb)); Qloss=sum(Qf[l]+Qt[l] for l in range(nb))
    obj=(a_d*cp.square(Pd)+b_d*Pd)*dt+cP*Ploss*dt+cQ*Qloss*dt+c_eq*zeq
    pr=cp.Problem(cp.Minimize(obj),C); pr.solve(solver='CLARABEL',verbose=verbose)
    ci,cj,sj=cii.value,cij.value,sij.value
    gap=[ci[idx[f]]*ci[idx[t]]-cj[l]**2-sj[l]**2 for l,(f,t,*_) in enumerate(br)]
    relgap=[gap[l]/(ci[idx[f]]*ci[idx[t]]) for l,(f,t,*_) in enumerate(br)]
    # angle recovery along tree from 650
    th=np.zeros(N)
    for l,(f,t,*_) in enumerate(br): th[idx[t]]=th[idx[f]]-np.arctan2(sj[l],cj[l])
    V=np.sqrt(ci)
    # AC power flow check: recompute flows with recovered phasors
    Vc=V*np.exp(1j*th); mism=[]
    for l,(f,t,*_) in enumerate(br):
        Sf=Vc[idx[f]]*np.conj((Vc[idx[f]]-Vc[idx[t]])*y[l]); mism.append(abs(Sf-(Pf[l].value+1j*Qf[l].value)))
    served=sum(x.value[idx[k]]*PL[k] for k in buses)
    return dict(status=pr.status,obj=pr.value,Pd=float(Pd.value),Qd=float(Qd.value),Ploss=float(Ploss.value),Qloss=float(Qloss.value),
        zeq=float(zeq.value),x={k:x.value[idx[k]] for k in buses if PL[k]>0},V=dict(zip(buses,V)),
        th=dict(zip(buses,np.degrees(th))),maxgap=max(np.abs(gap)),maxrel=max(np.abs(relgap)),mism=max(mism),
        served=served,total=sum(PL.values()),ppv={k:ppv[k].value for k in PV},pbs={k:pbs[k].value for k in BESS},
        flows={f"{f}-{t}":(Pf[l].value,Qf[l].value,np.hypot(Pf[l].value,Qf[l].value)) for l,(f,t,*_) in enumerate(br)},
        time=pr.solver_stats.solve_time, nvar=sum(v.size for v in pr.variables()), ncon=len(C))
if __name__=='__main__':
    for ce in [500,2000,8000]:
        r=solve(ce)
        print('c_eq',ce,r['status'],'obj',round(r['obj'],2),'Pd',round(r['Pd'],4),'Qd',round(r['Qd'],4),
              'Ploss kW',round(r['Ploss']*1e3,2),'Qloss kvar',round(r['Qloss']*1e3,2),'z',round(r['zeq'],4),
              'served',round(r['served'],4),'/',round(r['total'],4),'gap',r['maxgap'],r['maxrel'],'mism',r['mism'],'t',r['time'])
        print(' x',{k:round(v,4) for k,v in r['x'].items()})
        print(' V',{k:round(v,4) for k,v in r['V'].items()}); print(' th',{k:round(v,3) for k,v in r['th'].items()})
        print(' pv',r['ppv'],'bess',r['pbs']); print(' flows',{k:round(v[2],3) for k,v in r['flows'].items()})
    print(r['nvar'],r['ncon'])
