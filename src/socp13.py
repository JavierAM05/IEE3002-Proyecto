# LEAN LOS COMENTARIOS :D
import numpy as np
import cvxpy as cp

# Primero definimos los parametros 

Sb = 1.0 # potencia base
Vb = 4.16 # voltaje base
Zb = Vb**2 / Sb # impedancia base
mi = 5280.0 # millas

# Impedancias de las ramas 
z = {
    '601': (0.3465, 1.0179),
    '602': (0.7526, 1.1814),
    '603': (1.3294, 1.3471),
    '604': (1.3238, 1.3569),
    '605': (1.3292, 1.3475),
    '606': (0.7982, 0.4463),
    '607': (1.3425, 0.5124)
}

# Ramas del sistema: (nodo origen, nodo destino , tipo de cable, longitud en pies, Smax_MVA)
br = [
    ('650', '632', '601', 2000, 2.5),
    ('632', '633', '602', 500,  1.0),
    ('633', '634', 'XF',  0,   0.5), 
    ('632', '645', '603', 500,  1.0),
    ('645', '646', '603', 300,  1.0),
    ('632', '671', '601', 2000, 2.5),
    ('671', '680', '601', 1000, 1.0),
    ('671', '684', '604', 300,  1.0),
    ('684', '611', '605', 300,  1.0),
    ('684', '652', '607', 800,  1.0),
    ('671', '692', 'SW',  0,   2.5),
    ('692', '675', '606', 500,  1.5)
]

# nodos y mapeo de indices
buses = ['650', '632', '633', '634', '645', '646', '671', '680', '684', '611', '652', '692', '675']
idx = {b: k for k, b in enumerate(buses)}
N = len(buses)

# Calculo para las matrices de impedancia y admitancia serie en por unidad
R, X = [], []
for f, t, c, L, _ in br:
    if c == 'XF':
        r, x = 0.011 * Sb / 0.5, 0.02 * Sb / 0.5
    elif c == 'SW':
        r, x = 1e-4, 1e-4
    else:
        r, x = z[c][0] * L / mi / Zb, z[c][1] * L / mi / Zb
    R.append(r)
    X.append(x)

R = np.array(R)
X = np.array(X)
y = 1.0 / (R + 1j * X) # Admitancia serie compleja
g = y.real
b = y.imag # Susceptancia de rama


# Aqui va la parte de la demandaa, y recursos 

# cargas nominales tipo P [MW], Q [Mvar] 
PL = dict.fromkeys(buses, 0.0)
QL = dict.fromkeys(buses, 0.0)
cargas_base = {
    '634': (0.400,0.290 ), 
    '645': (0.170, 0.125),
    '646': (0.230, 0.132),
    '652':(0.128, 0.086),
    '671': (1.155 + 0.200, 0.660 + 0.116),
    '675': (0.843, 0.462),
    '692':  (0.170, 0.151),
    '611': (0.170, 0.080)
}
for k, (p, q) in cargas_base.items():
    PL[k] = p
    QL[k] = q

# Ponderadores de prioridad social de racionamiento w_i 
w = dict.fromkeys(buses, 1.0)
w['671'] = 3.0  # Hospital / servicio de salud
w['652'] = 2.0 # Sistema de bombeo de agua potable

# Recursos distribuidos (inversores PV y batería BESS)
PV = {'675': 0.45, '680': 0.30}  # Capacidad máxima activa P_pv 
PVq = 0.20 # Margen para soporte de reactivos Q_pv
BESS = {'671': 0.25} # Descarga de batería P_bess

# Parametros del generador diesel de emergencia 
Pd_max, Qd_max = 2.0, 1.5 # Capacidades máximas en [MW] y [Mvar]
a_d, b_d = 20.0, 230.0 # Costo cuadrático [USD/MW^2-h] y lineal [USD/MWh]
cP, cQ = 150.0, 30.0 # Costos de pérdidas activas y reactivas [USD/MWh], [USD/Mvarh]
dt = 1.0 # Paso temporal 


# vemos la función de resolución del modelo SOCP
def solve(c_eq, verbose=False):
    nb = len(br)

    # variables de red de Jabr
    cii = cp.Variable(N)
    cij = cp.Variable(nb)
    sij = cp.Variable(nb)

    # Variables de racionamiento y equidad
    x = cp.Variable(N) # Fracción de carga conectada 
    zeq = cp.Variable() # Variable epigráfica

    # Variables de despacho
    Pd = cp.Variable()
    Qd = cp.Variable()
    ppv = {k: cp.Variable() for k in PV}
    qpv = {k: cp.Variable() for k in PV}
    pbs = {k: cp.Variable() for k in BESS}

    # Flujos de potencia en las ramas 
    Pf, Pt, Qf, Qt = [], [], [], []
    for l, (f, t, *_) in enumerate(br):
        i, j = idx[f], idx[t]
        Pf.append(g[l] * (cii[i] - cij[l]) - b[l] * sij[l]) # Flujo P desde i hacia j
        Qf.append(-b[l] * (cii[i] - cij[l]) - g[l] * sij[l]) # Flujo Q desde i hacia j
        Pt.append(g[l] * (cii[j] - cij[l]) + b[l] * sij[l]) # Flujo P desde j hacia i
        Qt.append(-b[l] * (cii[j] - cij[l]) + g[l] * sij[l]) # Flujo Q desde j hacia i

    # restricciones basicas de la operacion
    C = [
        0.95**2 <= cii, cii <= 1.05**2,
        0 <= x, x <= 1,
        0 <= Pd, Pd <= Pd_max,
        cp.abs(Qd) <= Qd_max
    ]

    # restriccionescónicas de segundo orden (SOCP)
    for l, (f, t, *_) in enumerate(br):
        i, j = idx[f], idx[t]
        # Cono de Lorentz rotado
        C += [cp.SOC(cii[i] + cii[j], cp.hstack([2 * cij[l], 2 * sij[l], cii[i] - cii[j]]))]
        
        # Limites de capacidad termica de las ramas
        Smax = br[l][4]
        C += [
            cp.SOC(Smax, cp.hstack([Pf[l], Qf[l]])),
            cp.SOC(Smax, cp.hstack([Pt[l], Qt[l]]))
        ]

    # Capacidad conica de inversores fotovoltaicos
    for k in PV:
        C += [
            0 <= ppv[k], ppv[k] <= PV[k],
            cp.SOC(np.hypot(PV[k], PVq), cp.hstack([ppv[k], qpv[k]]))
        ]

    # Limites de potencia activa 
    for k in BESS:
        C += [0 <= pbs[k], pbs[k] <= BESS[k]]

    # Balances nodales de potencia 
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

        fp = sum(Pf[l] for l, (f, t, *_) in enumerate(br) if f == bn) + \
             sum(Pt[l] for l, (f, t, *_) in enumerate(br) if t == bn)
        fq = sum(Qf[l] for l, (f, t, *_) in enumerate(br) if f == bn) + \
             sum(Qt[l] for l, (f, t, *_) in enumerate(br) if t == bn)

        C += [fp == Pinj, fq == Qinj]

        # Formulación epigrafica de la equidad rawlsiana
        if PL[bn] > 0:
            C += [zeq >= w[bn] * (1 - x[i]) * dt]
        else:
            C += [x[i] == 1]

    # Perdidas totales del sistema
    Ploss = sum(Pf[l] + Pt[l] for l in range(nb))
    Qloss = sum(Qf[l] + Qt[l] for l in range(nb))

    # Función objetivo multiobjetivo: Combustible + Pérdidas + Penalización por equidad
    obj = (a_d * cp.square(Pd) + b_d * Pd) * dt + cP * Ploss * dt + cQ * Qloss * dt + c_eq * zeq
    pr = cp.Problem(cp.Minimize(obj), C)
    pr.solve(solver='CLARABEL', verbose=verbose)

    # Post-procesamiento y verificación de exactitud
    ci, cj, sj = cii.value, cij.value, sij.value

    # Brechas del cono (gap absoluto y relativo)
    gap = [ci[idx[f]] * ci[idx[t]] - cj[l]**2 - sj[l]**2 for l, (f, t, *_) in enumerate(br)]
    relgap = [gap[l] / (ci[idx[f]] * ci[idx[t]]) for l, (f, t, *_) in enumerate(br)]

    # Recuperación fasorial de angulos a lo largo  arbol desde la subestación 
    th = np.zeros(N)
    for l, (f, t, *_) in enumerate(br):
        th[idx[t]] = th[idx[f]] - np.arctan2(sj[l], cj[l])
    V = np.sqrt(ci)

    # comprobación de factibilidad física AC con fasores recuperados
    Vc = V * np.exp(1j * th)
    mism = []
    for l, (f, t, *_) in enumerate(br):
        Sf = Vc[idx[f]] * np.conj((Vc[idx[f]] - Vc[idx[t]]) * y[l])
        mism.append(abs(Sf - (Pf[l].value + 1j * Qf[l].value)))

    served = sum(x.value[idx[k]] * PL[k] for k in buses)

    return dict(
        status=pr.status, obj=pr.value, Pd=float(Pd.value), Qd=float(Qd.value),
        Ploss=float(Ploss.value), Qloss=float(Qloss.value), zeq=float(zeq.value),
        x={k: x.value[idx[k]] for k in buses if PL[k] > 0}, V=dict(zip(buses, V)),
        th=dict(zip(buses, np.degrees(th))), maxgap=max(np.abs(gap)),
        maxrel=max(np.abs(relgap)), mism=max(mism), served=served,
        total=sum(PL.values()), ppv={k: ppv[k].value for k in PV},
        pbs={k: pbs[k].value for k in BESS},
        flows={f"{f}-{t}": (Pf[l].value, Qf[l].value, np.hypot(Pf[l].value, Qf[l].value)) for l, (f, t, *_) in enumerate(br)},
        time=pr.solver_stats.solve_time, nvar=sum(v.size for v in pr.variables()), ncon=len(C)
    )


# Ejecución (barrido de valores de c_eq) y resultados
if __name__ == '__main__':
    valores_ceq = [400, 600, 800, 1000, 1500, 8000]
    print(f"{'c_eq':>6} | {'Pd [MW]':>8} | {'Abast [MW]':>10} | {'Ploss [kW]':>10} | {'z [h]':>8} | {'Vmin [pu]':>9} | {'Max Gap':>10}")
    print("-" * 75)
    for ce in valores_ceq:
        r = solve(ce)
        vmin = min(r['V'].values())
        print(f"{ce:6d} | {r['Pd']:8.3f} | {r['served']:10.3f} | {r['Ploss']*1e3:10.2f} | {r['zeq']:8.3f} | {vmin:9.3f} | {r['maxrel']:.2e}")