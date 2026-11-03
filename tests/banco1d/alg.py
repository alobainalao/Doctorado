"""Comprueba, en precisión de máquina, las recursiones del adjunto discreto del documento (ADR, bloque, semi)
con matrices densas aleatorias: dJ/dF^n debe ser -psi_C^{n+1} (convención L = J + sum psi^T residuo)."""
import numpy as np
rng=np.random.default_rng(0); N=6; Nt=4; Nr=2
Pm=[np.eye(N)*3+0.3*rng.standard_normal((N,N)) for _ in range(Nt+2)]   # P^{n}, n=1..Nt
Rm=[0.8*np.eye(N)+0.2*rng.standard_normal((N,N)) for _ in range(Nt+2)] # R^{n}, n=0..Nt-1
G=np.diag(rng.uniform(0.5,2,N)); gam=1.3; dt=0.7
kap=rng.uniform(0.05,0.3,Nr); Qd=[np.diag(rng.uniform(0.1,0.5,N)) for _ in range(Nr)]
ee=rng.uniform(0.3,0.9,Nr); aa=rng.uniform(0.1,0.4,Nr)
def jn(C): return gam*dt*C@G@C
def gjn(C): return 2*gam*dt*G@C
def fwd(model,F,C0,Cim0):
    C=[C0]; Cim=[Cim0]; Cs=[None]
    for n in range(Nt):
        if model=="adr":
            C.append(np.linalg.solve(Pm[n+1],Rm[n]@C[n]+F[n]))
        elif model=="block":
            rhs=Rm[n]@C[n]+sum(Qd[r]@Cim[n][r] for r in range(Nr))+F[n]; Cn=np.linalg.solve(Pm[n+1],rhs); C.append(Cn)
            Cim.append([kap[r]*(Cn+C[n])+(1-2*kap[r])*Cim[n][r] for r in range(Nr)])
        else:
            Cst=np.linalg.solve(Pm[n+1],Rm[n]@C[n]+F[n]); Cs.append(Cst); Ct=Cst.copy(); new=[]
            for r in range(Nr):
                new.append((1-ee[r])*Ct+ee[r]*Cim[n][r]); Ct=(1-aa[r])*Ct+aa[r]*Cim[n][r]
            C.append(Ct); Cim.append(new)
    J=sum(jn(C[n]) for n in range(1,Nt+1)); return J,C,Cim,Cs
def adj(model,C,Cim,Cs):
    psi={Nt+1:np.zeros(N)}; psr={Nt+1:[np.zeros(N)]*Nr, Nt:[np.zeros(N)]*Nr}
    if model=="adr":
        for n in range(Nt,0,-1): psi[n]=np.linalg.solve(Pm[n].T,Rm[n].T@psi[n+1]-gjn(C[n]))
    elif model=="block":
        for n in range(Nt,0,-1):
            psr[n]=[(1-2*kap[r])*psr[n+1][r]+Qd[r].T@psi[n+1] for r in range(Nr)]
            rhs=Rm[n].T@psi[n+1]+sum(kap[r]*(psr[n][r]+psr[n+1][r]) for r in range(Nr))-gjn(C[n])
            psi[n]=np.linalg.solve(Pm[n].T,rhs)
    else:
        for n in range(Nt-1,-1,-1):
            mu=Rm[n+1].T@psi[n+2]-gjn(C[n+1]) if n+1<=Nt else None
            mus=[None]*(Nr+1); mus[Nr]=mu; new=[None]*Nr
            for r in range(Nr,0,-1):
                mus[r-1]=(1-aa[r-1])*mus[r]+(1-ee[r-1])*psr[n+1][r-1]
                new[r-1]=ee[r-1]*psr[n+1][r-1]+aa[r-1]*mus[r]
            psi[n+1]=np.linalg.solve(Pm[n+1].T,mus[0]); psr[n]=new
    return psi
for model in ("adr","block","semi"):
    F=[rng.standard_normal(N) for _ in range(Nt)]; C0=rng.standard_normal(N); Cim0=[rng.standard_normal(N) for _ in range(Nr)]
    J,C,Cim,Cs=fwd(model,F,C0,Cim0); psi=adj(model,C,Cim,Cs); err=0
    for n in range(Nt):
        for i in range(N):
            e=1e-6; Fp=[f.copy() for f in F]; Fm=[f.copy() for f in F]; Fp[n][i]+=e; Fm[n][i]-=e
            fd=(fwd(model,Fp,C0,Cim0)[0]-fwd(model,Fm,C0,Cim0)[0])/(2*e); err=max(err,abs(fd-(-psi[n+1][i])))
    print(f"{model:6s}: max|dJ/dF - (-psi)| = {err:.2e}")
