"""
Balance de agua del pozo: acopla la extracción Q(t) a un tanque de
almacenamiento del que se consume a una tasa diaria fija, con capacidad máxima.

Motivación: sin esto, el bombeo Q es SÓLO un costo en el funcional, así que el
óptimo trivial es Q→0 (no extraer). Físicamente el pozo existe para ABASTECER
agua, así que la extracción debe cubrir un consumo; el almacenamiento amortigua
la diferencia. Esto vuelve la extracción obligatoria y convierte el problema en
"extraer lo necesario minimizando contaminación y energía".

Modelo (tanque, por paso de tiempo n = 0..Nt−1):

    S_n = S0 + Vscale·dt·Σ_{k≤n} Q_k  −  cons_step·(n+1)

con cons_step = demand_day·dt/86400 (consumo por paso; demand_day es volumen/día
y dt está en segundos). Restricciones DURAS, lineales en Q:

    0 ≤ S_n ≤ S_max      (no vaciarse ⇒ suministro mínimo garantizado;
                          no rebosar ⇒ capacidad máxima)

Este módulo sólo usa numpy/scipy (sin rbf ni dolfinx), así que lo comparten los
dos backends (view/animation.py:optimize_bfr y fenicsx/adjoint.py:optimize_fenicsx).

Parámetros (config `p`, controlables desde la app):
  Vscale      conversión de la fuerza del sumidero Q a caudal de volumen
  demand_day  consumo (volumen/día)
  S0          almacenamiento inicial
  S_max       capacidad máxima del tanque
"""
import numpy as np

_SECONDS_PER_DAY = 86400.0


def supply_params(p):
    """Lee los parámetros de suministro de `p` con defaults, ya convertidos a
    magnitudes por paso de tiempo."""
    dt = float(p.dt)
    return dict(
        Vscale=float(getattr(p, "Vscale", 1.0)),
        cons_step=float(getattr(p, "demand_day", 120.0)) * dt / _SECONDS_PER_DAY,
        S0=float(getattr(p, "S0", 0.0)),
        S_max=float(getattr(p, "S_max", 500.0)),
        dt=dt,
    )


def storage_trajectory(Q, p):
    """Trayectoria del almacenamiento S_n para el control Q (array (Nt,))."""
    sp = supply_params(p)
    Q = np.asarray(Q, float)
    n = np.arange(1, len(Q) + 1)
    inflow = sp["Vscale"] * sp["dt"] * np.cumsum(Q)
    return sp["S0"] + inflow - sp["cons_step"] * n


def supply_constraint(p, n_controls):
    """Devuelve (A, lb, ub) de la restricción lineal 0 ≤ S ≤ S_max sobre el
    vector de controles x = [Q_0..Q_{Nt−1}, z_p] (longitud n_controls = Nt+1).

    S_n = A·x + b con b_n = S0 − cons_step·(n+1); imponer 0 ≤ S_n ≤ S_max
    equivale a −b ≤ A·x ≤ S_max − b. La columna de z_p es 0 (el tanque no
    depende de la profundidad del pozo)."""
    sp = supply_params(p)
    Nt = n_controls - 1

    # A: triangular inferior de unos escalada (inflow acumulado), columna z_p = 0.
    A = np.zeros((Nt, n_controls))
    tri = np.tril(np.ones((Nt, Nt)))
    A[:, :Nt] = sp["Vscale"] * sp["dt"] * tri

    n = np.arange(1, Nt + 1)
    b = sp["S0"] - sp["cons_step"] * n
    lb = -b
    ub = sp["S_max"] - b
    return A, lb, ub
