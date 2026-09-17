"""
Balance de agua del pozo con demanda variable en el tiempo y ciclos de bomba.

Modelo del tanque (paso n = 0..Nt−1):

    S_n = S0 + Vscale·dt·Σ_{k≤n} Q_k  −  Σ_{k≤n} d_k

donde d_k es la demanda en el paso k, calculada con un perfil diurno que combina
consumo doméstico y riego agrícola.  La restricción DURA lineal (usable en
scipy.optimize.LinearConstraint) sigue siendo:

    0 ≤ S_n ≤ S_max

Los ciclos de operación de la bomba (período de encendido/reposo propio de
motores sumergibles) se modelan como cotas superiores variables en Q_n:

    0 ≤ Q_n ≤ Q_max · avail_n

donde avail_n ∈ [0,1] es la disponibilidad de la bomba en el intervalo n.

──────────────────────────────────────────────────────────────────────────────
Curvas de referencia
──────────────────────────────────────────────────────────────────────────────
Curva doméstica horaria:
  AWWA Manual M32 (2004), Distribution System Requirements for Fire Protection;
  Mays, L.W. (2011), Water Resources Engineering, 2ª ed., Wiley, Tabla 4-2;
  Alvarado-Granados et al. (2023), "Coefficients and curves of hourly and daily
  variations of water demand", Water Practice & Technology 18(8):1991–2009.
  doi:10.2166/wpt.2023.120

Curva de riego agrícola:
  Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998), Crop
  Evapotranspiration, FAO Irrigation and Drainage Paper No. 56, FAO Roma.
  Cap. 5 (demanda diaria); distribución horaria: riego por goteo, zona
  semiárida (sin riego nocturno, pico en horas de máxima ET).

Disponibilidad de la bomba (ciclos operación/reposo):
  Driscoll, F.G. (1986), Groundwater and Wells, 2ª ed., Johnson Screens, Cap. 16;
  Grundfos (2019), Submersible Motor Guide MS/MMS (lit. 6511946);
  NEMA MG-1 (2021), Motors and Generators, NEMA Standards Publication.
  — Motores de giro continuo (pozos profundos): sin límite de horas de marcha
    continua, pero se recomienda periodo de reposo nocturno para disminuir el
    desgaste térmico y facilitar mantenimiento preventivo.
  — Arranques máximos recomendados: 6/h para 50-100 HP; mínimo 60 s off.
  — A dt=10 h el ciclo sub-horario es transparente; la disponibilidad diaria
    (tabla por hora) refleja el contrato operativo y los descansos programados.

Unidades físicas
──────────────────────────────────────────────────────────────────────────────
El modelo de flujo BFR es 2D (sección x–z).  Q [m²/s] es la extracción por
unidad de longitud en la dirección perpendicular al plano (y).  Para obtener
el caudal volumétrico real se multiplica por Vscale = L_perp [m]:

    Q_vol [m³/s] = Q [m²/s] × Vscale [m]

Vscale = grid spacing = 30 m (anchura de una "rebanada" de la sección), lo que
da Q_vol_típico = 1e-3 × 30 = 0.03 m³/s = 30 L/s (pozo municipal profundo).
S_n, S0, S_max y d_n están en [m³] (volumétricos).

Parámetros en la configuración `p`:
  Vscale        L_perp [m], default 30.0  (= grid spacing)
  demand_day    doméstico   [m³/día], default 750.0  (5 000 hab × 150 L/día)
  demand_agri   riego       [m³/día], default 250.0
  S0            almacenamiento inicial [m³], default 500.0
  S_max         capacidad máxima [m³],      default 2000.0  (≈2 días)
  t0_hour       hora solar del inicio, default 6.0
  pump_cycling  bool — activa disponibilidad variable de la bomba, default True
  dt            paso de tiempo (s)
"""
import numpy as np

_SECONDS_PER_DAY = 86400.0

# ── Curva doméstica horaria ────────────────────────────────────────────────────
# Factores relativos a la media diaria (media = 1). Horas 0-23.
# Pico matutino ≈ 07h (factor 1.80), pico vespertino ≈ 18h (1.70),
# mínimo nocturno ≈ 03h (0.17).
# Fuente: AWWA M32 / Mays (2011) Tabla 4-2 / Alvarado-Granados et al. (2023)
_DOMESTIC_RAW = np.array([
    0.25, 0.20, 0.18, 0.17, 0.20, 0.40,   # 00-05
    1.10, 1.80, 1.60, 1.30, 1.10, 1.00,   # 06-11
    0.90, 0.85, 0.90, 1.00, 1.10, 1.30,   # 12-17
    1.70, 1.60, 1.30, 1.00, 0.70, 0.40,   # 18-23
], dtype=float)
_DOMESTIC_HOURLY = _DOMESTIC_RAW / _DOMESTIC_RAW.mean()  # media exactamente 1

# ── Curva de riego agrícola horaria ───────────────────────────────────────────
# Riego por goteo en zona semiárida: sin riego nocturno (00-05h), rampa en
# horas de máxima ET, cierre al atardecer (>19h).
# Fuente: Allen et al. (1998) FAO-56, distribución horaria derivada de
# evapotranspiración de referencia (ET₀ horaria, Cap. 5).
_AGRI_RAW = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # 00-05 sin riego
    0.5, 1.0, 1.2, 1.3, 1.4, 1.5,    # 06-11 inicio
    1.5, 1.5, 1.4, 1.3, 1.2, 1.0,    # 12-17 pico
    0.8, 0.5, 0.3, 0.0, 0.0, 0.0,    # 18-23 cierre
], dtype=float)
_agri_mean = _AGRI_RAW.mean()
_AGRI_HOURLY = _AGRI_RAW / _agri_mean if _agri_mean > 0 else _AGRI_RAW.copy()

# ── Disponibilidad horaria de la bomba ────────────────────────────────────────
# Factor en [0, 1]: fracción de la hora en que la bomba puede estar activa.
# Noche (00-05h): operación reducida al 30-40 % — reposo térmico del motor,
# mantenimiento preventivo programado (Grundfos MS/MMS).
# Día (06-21h): disponibilidad plena (1.0).
# Tarde-noche (22-23h): reducción progresiva.
# Fuente: Driscoll (1986) Cap. 16; Grundfos (2019) Motor Guide lit. 6511946;
#         NEMA MG-1 (2021) — límite 6 arranques/h para 50-100 HP.
_PUMP_AVAIL = np.array([
    0.4, 0.3, 0.3, 0.3, 0.4, 0.6,    # 00-05 reposo nocturno
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,    # 06-11 operación plena
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,    # 12-17 operación plena
    1.0, 1.0, 1.0, 0.8, 0.6, 0.4,    # 18-23 reducción progresiva
], dtype=float)


# ── Utilidades ────────────────────────────────────────────────────────────────

def _hour_factor(arr, t_h):
    """Interpolación lineal circular en tabla horaria de 24 valores."""
    h = float(t_h) % 24.0
    i = int(h)
    frac = h - i
    return float(arr[i]) * (1.0 - frac) + float(arr[(i + 1) % 24]) * frac


def supply_params(p):
    """Parámetros escalares del tanque (sin el perfil de demanda)."""
    return dict(
        Vscale=float(getattr(p, "Vscale", 1.0)),
        S0=float(getattr(p, "S0", 0.0)),
        S_max=float(getattr(p, "S_max", 200.0)),
        dt=float(p.dt),
    )


def demand_vector(p, Nt):
    """Demanda por paso de tiempo [vol/paso] con perfil diurno.

    d_n = (f_dom(h_n)·D_dom + f_agri(h_n)·D_agri) · dt / 86400

    donde h_n es la hora solar del punto medio del paso n.

    Parámetros:
      demand_day   D_dom  [vol/día], default 80.0
      demand_agri  D_agri [vol/día], default 0.0
      t0_hour      hora solar del inicio, default 6.0
      dt           paso (s)
    """
    dt     = float(p.dt)
    D_dom  = float(getattr(p, "demand_day",  80.0))
    D_agri = float(getattr(p, "demand_agri",  0.0))
    t0_h   = float(getattr(p, "t0_hour",      6.0))

    d = np.empty(Nt)
    for n in range(Nt):
        h = t0_h + (n + 0.5) * dt / 3600.0
        d[n] = (_hour_factor(_DOMESTIC_HOURLY, h) * D_dom
                + _hour_factor(_AGRI_HOURLY, h) * D_agri) * dt / _SECONDS_PER_DAY
    return d


def pump_upper_bounds(p, Nt, Q_max):
    """Cota superior de Q_n según disponibilidad horaria de la bomba.

    Q_upper[n] = Q_max · avail(h_n)

    Si p.pump_cycling es False (o ausente), devuelve Q_max constante.

    Parámetros:
      pump_cycling  bool, default True
      t0_hour       hora solar del inicio, default 6.0
      dt            paso (s)
    """
    if not getattr(p, "pump_cycling", True):
        return np.full(Nt, float(Q_max))

    dt   = float(p.dt)
    t0_h = float(getattr(p, "t0_hour", 6.0))
    ub   = np.empty(Nt)
    for n in range(Nt):
        h = t0_h + (n + 0.5) * dt / 3600.0
        ub[n] = float(Q_max) * _hour_factor(_PUMP_AVAIL, h)
    return ub


def storage_trajectory(Q, p):
    """Trayectoria del almacenamiento S_n para el control Q (array (Nt,))."""
    sp  = supply_params(p)
    Q   = np.asarray(Q, float)
    d   = demand_vector(p, len(Q))
    return sp["S0"] + sp["Vscale"] * sp["dt"] * np.cumsum(Q) - np.cumsum(d)


def supply_constraint(p, n_controls):
    """Restricción lineal del balance del tanque: 0 ≤ S_n ≤ S_max.

    Balance por paso:  S_n = S_{n-1} + Vscale·dt·Q_n − d_n

    Esto implica:
      S_n ≥ 0  ↔  Vscale·dt·Q_n + S_{n-1} ≥ d_n
                   (extracción + almacenado ≥ demanda → suministro siempre cubierto)
      S_n ≤ S_max  ↔  Vscale·dt·Q_n ≤ d_n + (S_max − S_{n-1})
                   (nunca extraer más que demanda + espacio libre en el tanque)

    Vector de controles: x = [Q_0 .. Q_{Nt-1}, z_p]  (longitud n_controls = Nt+1).
    La columna de z_p en A es 0 (el tanque no depende de la profundidad del pozo)."""
    sp  = supply_params(p)
    Nt  = n_controls - 1

    A = np.zeros((Nt, n_controls))
    A[:, :Nt] = sp["Vscale"] * sp["dt"] * np.tril(np.ones((Nt, Nt)))

    d = demand_vector(p, Nt)
    b = sp["S0"] - np.cumsum(d)
    return A, -b, sp["S_max"] - b


def pump_duty_constraint(p, n_controls, Q_max):
    """Restricción de ciclo trabajo/descanso de la bomba (ventana deslizante).

    Modela: si la bomba opera pump_ton horas seguidas, debe descansar al menos
    pump_toff horas antes de volver a arrancar.

    En términos de la ventana de W pasos (donde W·dt ≥ pump_ton + pump_toff):

        Σ_{k=n}^{n+W-1} Q_k  ≤  f_duty · Q_max · W      ∀ n

    con f_duty = pump_ton / (pump_ton + pump_toff).

    La restricción es lineal en Q y se añade como segunda LinearConstraint.
    Si pump_cycling=False devuelve None (sin restricción de ciclo).

    Parámetros (en p):
      pump_ton    horas máx de operación continua, default 20.0
      pump_toff   horas mín de descanso tras pump_ton horas, default 4.0
      pump_cycling  bool — activa la restricción, default True
      dt          paso de tiempo (s)

    Fuente: Grundfos Submersible Motor Guide MS/MMS (2019);
            NEMA MG-1 (2021) — degradación térmica y límite de arranques.
    """
    if not getattr(p, "pump_cycling", True):
        return None

    dt   = float(p.dt)
    Nt   = n_controls - 1
    ton  = float(getattr(p, "pump_ton",  20.0))   # horas
    toff = float(getattr(p, "pump_toff",  4.0))   # horas
    T_cycle_s = (ton + toff) * 3600.0
    f_duty    = ton / (ton + toff)

    # Número de pasos que cubre un ciclo completo (mínimo 2)
    W = max(2, int(np.ceil(T_cycle_s / dt)))
    W = min(W, Nt)

    n_rows = Nt - W + 1
    if n_rows <= 0:
        return None

    A = np.zeros((n_rows, n_controls))
    for i in range(n_rows):
        A[i, i:i + W] = 1.0          # suma de W extracciones consecutivas

    ub = f_duty * float(Q_max) * W * np.ones(n_rows)
    lb = np.zeros(n_rows)             # trivial (Q ≥ 0 ya garantizado por bounds)
    return A, lb, ub
