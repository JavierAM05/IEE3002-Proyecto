import warnings
import cvxpy as cp
import numpy as np

# Importamos los parametros de la red, matrices de impedancia y cotaas desde el modelo base SOCP
from socp13 import *

warnings.filterwarnings('ignore')

# Declaramos las variables para el LINDISTFLOW

nb = len(br)

# Variables de flujo de rama y magnitudes nodales 
P = cp.Variable(nb) # Flujo de potencia P por cada rama
Q = cp.Variable(nb) # Flujo de potencia Q por cada rama
v = cp.Variable(N) # Cuadrado de la magnitud de voltaje nodal aproximada

# Variables de racionamiento y equidad social
x = cp.Variable(N) # Porción de carga conectada por nodo
zeq = cp.Variable() # Variable epigrafica minimax

# Variables de despacho de generación y almacenamiento
Pd = cp.Variable() # P del generador diesel en barra 650 
Qd = cp.Variable() # Q del generador diesel en barra 650 
ppv = {k: cp.Variable() for k in PV} # P despachada por inversor solar
qpv = {k: cp.Variable() for k in PV} # Q despachado por inversor solar 
pbs = {k: cp.Variable() for k in BESS} # P inyectada por batería BESS 

# formulacion y restricciones de LINDISTFLOW

# restricciones basicas de operacion 
C = [
    0.95**2 <= v, v <= 1.05**2,
    0 <= x, x <= 1,
    0 <= Pd, Pd <= Pd_max,
    cp.abs(Qd) <= Qd_max
]

# Capacidad aparente conico de los inversores solares
for k in PV:
    C += [
        0 <= ppv[k], ppv[k] <= PV[k],
        cp.SOC(np.hypot(PV[k], PVq), cp.hstack([ppv[k], qpv[k]]))
    ]

# Cota de descarga de baterias
for k in BESS:
    C += [0 <= pbs[k], pbs[k] <= BESS[k]]

# Ecuaciones caida de voltaje y limites temics por rama
for l, (f, t, c, L, S) in enumerate(br):
    # Caída de voltaje linealizada:
    C += [
        v[idx[t]] == v[idx[f]] - 2 * (R[l] * P[l] + X[l] * Q[l]),
        cp.SOC(S, cp.hstack([P[l], Q[l]])) # Capacidad termica aparente de la linea
    ]

# balance de nodos de potencia activa y reactiva 
for bn in buses:
    i = idx[bn]
    Pinj = -x[i] * PL[bn]
    Qinj = -x[i] * QL[bn]

    if bn == '650':
        Pinj += Pd
        Qinj += Qd
    if bn in PV:
        Pinj += ppv[bn]
        Qinj += qpv[bn]
    if bn in BESS:
        Pinj += pbs[bn]

    # suma de flujos salientes menos flujos entrantes
    out = sum(P[l] for l, (f, *_) in enumerate(br) if f == bn)
    inn = sum(P[l] for l, (f, t, *_) in enumerate(br) if t == bn)
    outq = sum(Q[l] for l, (f, *_) in enumerate(br) if f == bn)
    innq = sum(Q[l] for l, (f, t, *_) in enumerate(br) if t == bn)

    C += [out - inn == Pinj, outq - innq == Qinj]

    # penalización minimax ponderada por prioridad social
    if PL[bn] > 0:
        C += [zeq >= w[bn] * (1 - x[i])]
    else:
        C += [x[i] == 1]

# Resolución del modelo LINDISTFLOW

# Función objetivo: Costo de diesel + penalización de equidad
pr = cp.Problem(cp.Minimize(a_d * cp.square(Pd) + b_d * Pd + 8000 * zeq), C)
pr.solve(solver='CLARABEL')

# Resultados obtenidos con la aproximación lineal
print(pr.status, 'Pd', Pd.value, 'Qd', Qd.value, 'z', zeq.value, 'served', sum(x.value[idx[k]] * PL[k] for k in buses))
print('V lin', {b: round(float(np.sqrt(v.value[idx[b]])), 4) for b in buses})

# Almacenamos el plan de racionamiento lineal para la verificación de factibilidad fisica
xs = x.value.copy()

# Validacion en red física real con el modelo SOCP
print("\n--- Ejecutando AC Feasibility Check para LinDistFlow ---")

# Declaracion de variables exactas del modelo SOCP de Jabr
cii_ac = cp.Variable(N) 
cij_ac = cp.Variable(nb)
sij_ac = cp.Variable(nb)  

Pd_ac = cp.Variable()
Qd_ac = cp.Variable()
ppv_ac = {k: cp.Variable() for k in PV}
qpv_ac = {k: cp.Variable() for k in PV}
pbs_ac = {k: cp.Variable() for k in BESS}

# Flujos de P y Q exactos en ambos extremos de cada rama
Pf_ac, Pt_ac, Qf_ac, Qt_ac = [], [], [], []
for l, (f, t, *_) in enumerate(br):
    i, j = idx[f], idx[t]
    Pf_ac.append(g[l] * (cii_ac[i] - cij_ac[l]) - b[l] * sij_ac[l]) # Flujo P salida 
    Qf_ac.append(-b[l] * (cii_ac[i] - cij_ac[l]) - g[l] * sij_ac[l]) # Flujo Q salida 
    Pt_ac.append(g[l] * (cii_ac[j] - cij_ac[l]) + b[l] * sij_ac[l]) # Flujo P llegada 
    Qt_ac.append(-b[l] * (cii_ac[j] - cij_ac[l]) + g[l] * sij_ac[l])# Flujo Q llegada 

# Limites de voltaje y generador de respaldo en el modelo AC
C_ac = [
    0.95**2 <= cii_ac, cii_ac <= 1.05**2,
    0 <= Pd_ac, Pd_ac <= Pd_max,
    cp.abs(Qd_ac) <= Qd_max
]

# Restricciones de capacidad de los inversores solares y baterías
for l, (f, t, *_) in enumerate(br):
    i, j = idx[f], idx[t]
    # Cono de Lorentz
    C_ac += [cp.SOC(cii_ac[i] + cii_ac[j], cp.hstack([2 * cij_ac[l], 2 * sij_ac[l], cii_ac[i] - cii_ac[j]]))]

    # Capacidad termica aparente en los dos terminales de la linea
    Smax = br[l][4]
    C_ac += [
        cp.SOC(Smax, cp.hstack([Pf_ac[l], Qf_ac[l]])),
        cp.SOC(Smax, cp.hstack([Pt_ac[l], Qt_ac[l]]))
    ]

# Restricciones de capacidad de recursos distribuidos
for k in PV:
    C_ac += [
        0 <= ppv_ac[k], ppv_ac[k] <= PV[k],
        cp.SOC(np.hypot(PV[k], PVq), cp.hstack([ppv_ac[k], qpv_ac[k]]))
    ]
for k in BESS:
    C_ac += [0 <= pbs_ac[k], pbs_ac[k] <= BESS[k]]

# Balances nodales verificando el plan de racionamiento calculado por LinDistFlow 
for bn in buses:
    i = idx[bn]
    Pinj = -xs[i] * PL[bn] # Demanda activa neta fijada por LinDistFlow
    Qinj = -xs[i] * QL[bn] # Demanda reactiva neta fijada por LinDistFlow

    if bn == '650':
        Pinj += Pd_ac
        Qinj += Qd_ac
    if bn in PV:
        Pinj += ppv_ac[bn]
        Qinj += qpv_ac[bn]
    if bn in BESS:
        Pinj += pbs_ac[bn]

    fp = sum(Pf_ac[l] for l, (f, t, *_) in enumerate(br) if f == bn) + \
         sum(Pt_ac[l] for l, (f, t, *_) in enumerate(br) if t == bn)
    fq = sum(Qf_ac[l] for l, (f, t, *_) in enumerate(br) if f == bn) + \
         sum(Qt_ac[l] for l, (f, t, *_) in enumerate(br) if t == bn)

    C_ac += [fp == Pinj, fq == Qinj]

# Resolver subproblema de factibilidad minimizando el despacho del diesel
prob_ac_check = cp.Problem(cp.Minimize(Pd_ac), C_ac)
prob_ac_check.solve(solver='CLARABEL')

# Verificación de factibilidad física 
print(f"Estado de factibilidad AC: {prob_ac_check.status}")
if prob_ac_check.status not in ["optimal", "optimal_inaccurate"]:
    print("Conclusión: El despacho de LinDistFlow es INFACTIBLE en la red AC real.")