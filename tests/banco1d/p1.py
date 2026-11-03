"""Banco 1D, Galerkin P1 con cuadratura nodal. Transporte con flujo dispersivo nulo (Neumann natural) en ambos
extremos (como el código); h con carga fija. Adjunto del flujo ensamblado con las fórmulas del documento
(b = psi C' + d, fuente -S d/dt(C psi)); NO se obtiene derivando las matrices discretas."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla

def cfg(**kw):
    c=dict(L=1.0,K=1.0,S=1.0,phi=1.0,R=1.0,D0=0.02,aL=0.05,lam0=0.1,T=0.5,n=200,nt=200,xp=0.4,eps=0.06,
           gamma=50.0,beta=0.0,dp=1.0,Nr=0,lam=[0.6,0.2],phim=[0.4,0.3],fa=1.0,theta=1.0,drop_d=False,drop_alm=False,nbc="N")
    c.update(kw); return c

def mesh(c):
    x=np.linspace(0,c['L'],c['n']+1); dx=x[1]-x[0]; m=np.full(len(x),dx); m[0]=m[-1]=dx/2; return x,dx,m

def gk(x,xp,eps): return np.exp(-(x-xp)**2/(2*eps**2))/np.sqrt(2*np.pi*eps**2)

def stiff(coef_e,dx,n):          # ∫ coef_e v' w'
    a=coef_e/dx; N=n+1
    main=np.zeros(N); main[:-1]+=a; main[1:]+=a
    off=-a
    return sp.diags([off,main,off],[-1,0,1],format='csc')

def adv(u_e,n):                  # ∫ u_e C' v  (fila i, columna j)
    N=n+1; rows=[];cols=[];vals=[]
    for sgn_row,(ri) in ((0,0),(1,1)):
        pass
    i=np.arange(n)
    # fila i (nodo izq): u/2 (C_{i+1}-C_i);  fila i+1: u/2 (C_{i+1}-C_i)
    r=np.concatenate([i,i,i+1,i+1]); cc=np.concatenate([i+1,i,i+1,i]); v=np.concatenate([u_e/2,-u_e/2,u_e/2,-u_e/2])
    return sp.csc_matrix((v,(r,cc)),shape=(N,N))

def dirich(A,rhs=None):
    A=A.tolil(); A[0,:]=0;A[-1,:]=0;A[0,0]=1;A[-1,-1]=1
    if rhs is not None: rhs[0]=rhs[-1]=0
    return A.tocsc()

def elem_u(h,c,dx): return -c['K']*np.diff(h)/dx

def forward(c,Q,xp=None):
    x,dx,m=mesh(c); n=c['n']; N=n+1; nt=c['nt']; dt=c['T']/nt; xp=c['xp'] if xp is None else xp
    g=gk(x,xp,c['eps']); h=np.zeros((nt+1,N)); C=np.zeros((nt+1,N)); Cim=np.zeros((nt+1,c['Nr'],N)); Ue=np.zeros((nt+1,n))
    C[0]=np.exp(-(x-0.7)**2/(2*0.08**2))
    Kmat=stiff(np.full(n,c['K']),dx,n); Mh=sp.diags(c['S']/dt*m,0,format='csc')
    for k in range(nt):
        A=dirich(Mh+Kmat); rhs=c['S']/dt*m*h[k]-Q[k+1]*g*m; rhs[0]=rhs[-1]=0
        h[k+1]=sla.spsolve(A,rhs); Ue[k+1]=elem_u(h[k+1],c,dx); ht=(h[k+1]-h[k])/dt; D_e=c['D0']+c['aL']*np.abs(Ue[k+1])
        c0=c['lam0'] if c['Nr']==0 else 0.0
        M=sp.diags(c['phi']*c['R']/dt*m,0)+stiff(D_e,dx,n)+adv(Ue[k+1],n)+sp.diags(m*(c0-c['fa']*c['S']*ht),0)
        rhs=c['phi']*c['R']/dt*m*C[k]
        for r in range(c['Nr']):
            l,pm=c['lam'][r],c['phim'][r]; q=(pm/dt)/(pm/dt+l)
            M=M+sp.diags(m*l*q,0); rhs=rhs+m*l*q*Cim[k,r]
        C[k+1]=sla.spsolve(M.tocsc(),rhs)
        for r in range(c['Nr']):
            l,pm=c['lam'][r],c['phim'][r]; Cim[k+1,r]=(pm/dt*Cim[k,r]+l*C[k+1])/(pm/dt+l)
    return dict(x=x,dx=dx,m=m,dt=dt,h=h,C=C,Cim=Cim,Ue=Ue,g=g)

def J_of(c,Q,xp=None):
    s=forward(c,Q,xp); nt=c['nt']; dt=s['dt']; w=np.full(nt+1,dt); w[0]=w[-1]=dt/2
    Jm=c['gamma']*sum(w[k]*np.sum(s['m']*s['g']*s['C'][k]**2) for k in range(nt+1))
    Jo=c['beta']*c['dp']*sum(w[k]*Q[k]**2 for k in range(nt+1)); return Jm+Jo,s

def adjoint(c,Q,s):
    x,dx,m,dt=s['x'],s['dx'],s['m'],s['dt']; n=c['n']; N=n+1; nt=c['nt']; g=s['g']; h,C,Ue=s['h'],s['C'],s['Ue']
    w=np.full(nt+1,dt); w[0]=w[-1]=dt/2
    psi=np.zeros((nt+2,N)); psih=np.zeros((nt+2,N)); psim=np.zeros((nt+2,c['Nr'],N)); Cp=np.zeros((nt+2,N))
    for k in range(nt,0,-1):
        u=Ue[k]; ht=(h[k]-h[k-1])/dt; D_e=c['D0']+c['aL']*np.abs(u); c0=c['lam0'] if c['Nr']==0 else 0.0
        # forma adjunta: a_C(eta,psi) = ∫D eta' psi' + ∫ u eta' psi + ∫ c eta psi  (transpuesta de la forma de estado)
        M=sp.diags(c['phi']*c['R']/dt*m,0)+stiff(D_e,dx,n)+adv(u,n).T+sp.diags(m*(c0-c['fa']*c['S']*ht),0)
        rhs=c['phi']*c['R']/dt*m*psi[k+1]-2*c['gamma']*(w[k]/dt)*m*g*C[k]
        for r in range(c['Nr']):
            l,pm=c['lam'][r],c['phim'][r]; q=(pm/dt)/(pm/dt+l)
            M=M+sp.diags(m*l*q,0); rhs=rhs+m*l*q*psim[k+1,r]
        psi[k]=sla.spsolve(M.tocsc(),rhs)
        for r in range(c['Nr']):
            l,pm=c['lam'][r],c['phim'][r]; psim[k,r]=(pm/dt*psim[k+1,r]+l*psi[k])/(pm/dt+l)
        Cp[k]=C[k]*psi[k]
    Kmat=stiff(np.full(n,c['K']),dx,n); Mh=sp.diags(c['S']/dt*m,0,format='csc')
    for k in range(nt,0,-1):
        u=Ue[k]; Cx=np.diff(C[k])/dx; px=np.diff(psi[k])/dx; pe=0.5*(psi[k][1:]+psi[k][:-1])
        d=(0.0 if c['drop_d'] else 1.0)*c['aL']*np.sign(u)*Cx*px; b=pe*Cx+d
        Fb=np.zeros(N); Fb[:-1]+=-c['K']*b; Fb[1:]+=c['K']*b           # ∫ K b v'
        alm=(0.0 if c['drop_alm'] else 1.0)*c['fa']*c['S']*m*((Cp[k+1]-Cp[k])/dt if k<nt else 0*x)
        rhs=c['S']/dt*m*psih[k+1]+Fb-alm; A=dirich(Mh+Kmat,rhs); psih[k]=sla.spsolve(A,rhs)
    G=np.zeros(nt+1)
    for k in range(1,nt+1): G[k]=2*c['beta']*c['dp']*Q[k]*w[k]+np.sum(m*g*psih[k])*dt
    return G,dict(psi=psi,psih=psih)
