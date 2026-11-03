import numpy as np
from p1 import *
def ctrl(c):
    t=np.linspace(0,c['T'],c['nt']+1); return 3.0*(1+0.5*np.sin(2*np.pi*t/c['T']*1.5)), np.cos(2*np.pi*t/c['T']*2)+0.5
def fdcheck(c,tag):
    Q,dQ=ctrl(c); J0,s=J_of(c,Q); G,_=adjoint(c,Q,s); e=1e-4
    fd=(J_of(c,Q+e*dQ)[0]-J_of(c,Q-e*dQ)[0])/(2*e); ad=float(G@dQ)
    print(f"  {tag:46s} n={c['n']:4d} adj={ad:+.5e} fd={fd:+.5e} err={abs(ad-fd)/abs(fd):.2e}")
for S in (1.0,4.0):
  print("S =",S,"(C con flujo dispersivo nulo en ambos extremos)")
  for n in (100,200,400):
    b=dict(S=S,n=n,nt=n)
    fdcheck(cfg(**b),"ADR, completo")
    fdcheck(cfg(**b,drop_alm=True),"ADR, sin fuente de almacenamiento en psi_h")
    fdcheck(cfg(**b,drop_d=True,aL=0.5),"ADR, sin termino d (aL=0.5)")
    fdcheck(cfg(**b,aL=0.5),"ADR, completo (aL=0.5)")
    fdcheck(cfg(**b,Nr=2),"MRMT Nr=2, completo")
    fdcheck(cfg(**b,fa=0.0),"modelo reducido (sin almacenamiento)")
