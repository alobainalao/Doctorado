import sys, json, numpy as np
sys.path.insert(0,"/home/alex/dev/doctorado/codigo-sumidero")
from scipy.spatial import cKDTree
from experimentos.animar import DOM_PATH
R="data/output/experimentos/"
def cargar(path):
    z=np.load(path,allow_pickle=True); nodes=np.asarray(z["nodes"],float)
    U=np.asarray(z["U"],float); U=U[:,0] if U.ndim==4 else U
    m=DOM_PATH.contains_points(nodes,radius=1e-3)
    H=np.asarray(z["H"],float)[:,0,m]; return nodes[m],H,U[:,m],float(z["dt"])
casos=[("H0_antiguo","bfr","fenicsx"),("H1_Q0p3","bfr","fenicsx"),("H2_malla_saneada","bfr","fenicsx"),("H3_r_pozo20","bfr","fenicsx"),
       ("H4_delta_area","bfr","fenicsx"),("H5_theta1","bfr","fenicsx"),("H6_bc_fem","bfr","H5_theta1")]
print(f"{'caso':18s} {'Hrms':>7s} {'Hmax':>7s} {'rangoFEM':>8s} {'Hrms/rg':>8s} {'dHpozo':>7s} {'|U|L2':>7s} {'|U|L2(t=1)':>10s}")
for c,b,f in casos:
    nb,Hb,Ub,dt=cargar(R+f"H_ablacion_flujo/{c}/bfr/simulation_results.npz")
    fp=R+f"H_ablacion_flujo/{c}/fenicsx/simulation_results.npz" if f=="fenicsx" else R+f"H_ablacion_flujo/{f}/fenicsx/simulation_results.npz"
    nf,Hf,Uf,_=cargar(fp)
    if len(nb)!=len(nf) or np.abs(nb-nf).max()>1e-6:   # nodos distintos: interpolar FEM a BFR (vecino más cercano)
        idx=cKDTree(nf).query(nb)[1]; Hf=Hf[:,idx]; Uf=Uf[:,idx]; nota="*"
    else: nota=""
    k=-1
    dH=Hb[k]-Hf[k]; Vb=np.linalg.norm(Ub,axis=-1); Vf=np.linalg.norm(Uf,axis=-1)
    kw=np.argmin(np.hypot(*(nb-np.array([5000.,-700.])).T)) if False else None
    rg=Hf[k].max()-Hf[k].min()
    l2=lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
    print(f"{c+nota:18s} {np.sqrt((dH**2).mean()):7.3f} {np.abs(dH).max():7.3f} {rg:8.3f} {np.sqrt((dH**2).mean())/rg:8.4f} {np.abs(Hb[k].min()-Hf[k].min()):7.3f} {l2(Vb[k],Vf[k]):7.3f} {l2(Vb[0],Vf[0]):10.3f}")
