"""
Catálogo de experimentos BFR (RBF-FD) vs FEniCS (FEM).

Cada grupo tiene `desc` y `pregunta` (la que responde, para el informe).
Cada caso es un conjunto de *cambios* respecto a `experimentos/base.json`
(malla compartida congelada, CI de funcion.h5, ADR, dt=18000 s, T=30 días).

Campos de un caso:
    desc       descripción corta (aparece en --list y en el resumen)
    cfg        cambios comunes a ambos backends
    bfr / fenicsx
               cambios sólo para ese backend
    backends   backends que se corren (default: ambos)
    ref_fem    "grupo/caso" cuyo resultado FEniCS sirve de referencia cuando
               el caso sólo cambia algo de BFR (evita repetir la corrida FEM)

Salidas: data/output/experimentos/<grupo>/<caso>/{bfr,fenicsx,comparacion}/
"""

BACKENDS = ("bfr", "fenicsx")
ENVS = {"bfr": "bfr_env", "fenicsx": "fenics_env"}

SIN_FUENTE = {"activate_fuente": False}
SIN_POZO = {"activate_ext": False}
T_W = 365 * 86400      # horizonte del grupo W (fuente apagada desde t = 30 d)
T_Z = 90 * 86400       # horizonte del grupo Z (limpieza completa del cuadrado)
C_Z = 5.94e-11         # C inicial del grupo Z: máximo de la CI del dominio real

GRUPOS = {
    # ── A. Términos fuente: aislar la contribución de fuente y pozo ──────────
    "A_terminos": {
        "desc": "Términos fuente: CI sola, sólo fuente, sólo pozo, ambos",
        "pregunta": "¿Cuánto de la diferencia BFR↔FEM aporta cada término (CI, fuente, pozo)?",
        "casos": {
            "A0_sin_fuente_sin_pozo": {
                "desc": "Sólo la CI transportada (sin fuente, sin pozo)",
                "cfg": {**SIN_FUENTE, **SIN_POZO},
            },
            "A1_solo_fuente": {
                "desc": "Fuente activa, pozo apagado",
                "cfg": {**SIN_POZO},
            },
            "A2_solo_pozo": {
                "desc": "Pozo activo (Q_std=0.3), fuente apagada",
                "cfg": {**SIN_FUENTE},
            },
            "A3_fuente_pozo": {
                "desc": "Fuente y pozo activos (configuración de referencia)",
                "cfg": {},
            },
        },
    },

    # ── B. BC de salida: BFR con la condición natural de FEniCS ──────────────
    # Sólo cambia BFR → la referencia FEM es el caso homólogo del grupo A.
    "B_bc_salida_fem": {
        "desc": "BFR con BC de salida tipo FEM (Neumann 1.er orden)",
        "pregunta": "¿La condición de salida de 2.º orden de BFR (pfs) explica la diferencia de flujo?",
        "casos": {
            f"B{i}_{nombre}": {
                "desc": f"Como A{i}, BFR con bc_outlet_mode=fem",
                "cfg": dict(cfg),
                "bfr": {"bc_outlet_mode": "fem"},
                "backends": ["bfr"],
                "ref_fem": f"A_terminos/A{i}_{nombre}",
            }
            for i, (nombre, cfg) in enumerate([
                ("sin_fuente_sin_pozo", {**SIN_FUENTE, **SIN_POZO}),
                ("solo_fuente", {**SIN_POZO}),
                ("solo_pozo", {**SIN_FUENTE}),
                ("fuente_pozo", {}),
            ])
        },
    },

    # ── C. Soporte de la fuente gaussiana (σ = 1 está en A1 / A3) ────────────
    "C_soporte_fuente": {
        "desc": "Ancho de la gaussiana de la fuente (sigma_fuente), con y sin pozo",
        "pregunta": "¿La diferencia en C viene de lo localizada que está la fuente?",
        "casos": {
            f"C_sigma{str(s).replace('.', 'p')}_{tag}": {
                "desc": f"sigma_fuente={s}, {'sin' if tag == 'sin_pozo' else 'con'} pozo",
                "cfg": {"sigma_fuente": s, **(SIN_POZO if tag == "sin_pozo" else {})},
            }
            for tag in ("sin_pozo", "con_pozo")
            for s in (0.5, 2.0, 4.0)
        },
    },

    # ── D. Caudal del pozo, sin fuente (Q_std = 0.3 está en A2) ──────────────
    "D_caudal_pozo": {
        "desc": "Caudal de extracción Q_std (cono de depresión), sin fuente",
        "pregunta": "¿Cómo crece la diferencia con el caudal (cono de depresión más fuerte)?",
        "casos": {
            f"D_Q{q:.0e}".replace("-0", "-"): {
                "desc": f"Q_std={q:g}, sin fuente",
                "cfg": {"Q_std": q, **SIN_FUENTE},
            }
            for q in (1e-2, 3e-2, 1e-1, 7e-1)
        },
    },

    # ── E. Paso de tiempo (dt = 18000 s está en A3) ──────────────────────────
    "E_paso_tiempo": {
        "desc": "Paso de tiempo dt, con fuente y pozo",
        "pregunta": "¿La diferencia es temporal (dt) o espacial/de formulación?",
        "casos": {
            f"E_dt{dt}": {"desc": f"dt={dt} s ({dt / 3600:g} h)", "cfg": {"dt": dt}}
            for dt in (36000, 7200)
        },
    },

    # ── F. Estabilización del transporte en BFR (upwind) ────────────────────
    "F_upwind_bfr": {
        "desc": "Upwind direccional en el transporte BFR",
        "pregunta": "¿El upwind direccional acerca el transporte BFR al de FEM?",
        "casos": {
            "F1_upwind_solo_fuente": {
                "desc": "Como A1, BFR con upwind_transport",
                "cfg": {**SIN_POZO},
                "bfr": {"upwind_transport": True},
                "backends": ["bfr"],
                "ref_fem": "A_terminos/A1_solo_fuente",
            },
            "F3_upwind_fuente_pozo": {
                "desc": "Como A3, BFR con upwind_transport",
                "cfg": {},
                "bfr": {"upwind_transport": True},
                "backends": ["bfr"],
                "ref_fem": "A_terminos/A3_fuente_pozo",
            },
        },
    },

    # ── W. Sumidero de transporte W- = χ(Q) g_ε C (decisión 1 de la tesis) ──
    # Fuente activa sólo los primeros 30 días; después no entra contaminante y
    # se sigue hasta T. Si la advección basta para sacar el soluto por el pozo,
    # el dominio se limpia también sin W-; balance_masa guarda cada término.
    "W_sumidero": {
        "desc": "Con y sin sumidero W- en el transporte; fuente apagada tras 30 días",
        "pregunta": "¿Hace falta W- para que el contaminante salga por el pozo?",
        "comparar": False,
        "casos": {
            f"W{int(con)}_{'con' if con else 'sin'}_sumidero": {
                "desc": f"{'Con' if con else 'Sin'} W-, fuente 30 d, T={T_W // 86400} d",
                "cfg": {"extract_C": con, "factor_sumidero_C": 1.0 if con else 0.0, "t_fuente_off": 2592000, "T": T_W,
                        "dt": 36000, "balance_masa": True},
                "backends": ("fenicsx",),
            }
            for con in (True, False)
        },
    },

    # ── Z. Cuadrado contaminado, sólo pozo (decisión 1, dinámica en el pozo) ─
    # Dominio cuadrado de 600 m, pozo en el centro, condición libre ∂_n(u·n) = 0
    # en todo el borde (modo "libre": el nodo de borde sigue la ecuación de
    # borde S_s ∂_t H = ∂_τ(K_τ ∂_τ H)), transporte con flujo dispersivo nulo en el borde,
    # C = 1 y u = 0 al inicio, sin fuente. Con C uniforme la forma no
    # conservativa sin W- debe conservar C = 1: el soluto sale con el agua.
    "Z_cuadrado": {
        "desc": "Cuadrado con C=1 inicial y pozo central; con y sin W-",
        "pregunta": "¿Sin W- el soluto sale por el pozo o se acumula?",
        "comparar": False,
        "casos": {
            f"Z{int(con)}_{'con' if con else 'sin'}_sumidero": {
                "desc": f"{'Con' if con else 'Sin'} W-, cuadrado, condición libre en todo el borde",
                "cfg": {"domain": "cuadrado", "mesh_pkl": None, "activate_fuente": False,
                        "extract_C": con, "factor_sumidero_C": 1.0 if con else 0.0, "bc_outlet_mode": "libre", "phi_const": 0.1,
                        "C_init_const": C_Z, "h_init_const": 300.0,
                        "lado_cuadrado": 600.0, "C_banda_limpia": 60.0, "T": T_Z,
                        "S_s_const": 1e-5, "Q_std": 0.01,
                        "spacing": 30, "min_sp": 4},
                "backends": ("bfr",),
            }
            for con in (True, False)
        },
    },

    # ── S. Sumidero W- del pozo (decisión 1) ─────────────────────────────────
    # W- = χ(Q) g_ε C escalado por factor_sumidero_C (0 por defecto desde la
    # comparativa del cuadrado, rama exp/sumidero-W). Configuración de
    # referencia ADR con el término (1) y sin él (0), en ambos backends.
    # Grilla 2×2 (backend × término): python -m experimentos.grilla_sumidero
    "S_sumidero": {
        "desc": "ADR de referencia con y sin el sumidero W- del pozo, BFR y FEniCS",
        "pregunta": "¿Cuánto cambia la solución (C, masa, C captada) al anular W-?",
        "casos": {
            "S1_con_termino": {"desc": "factor_sumidero_C = 1 (modelo anterior)",
                               "cfg": {"factor_sumidero_C": 1.0}},
            "S0_sin_termino": {"desc": "factor_sumidero_C = 0 (W- anulado)",
                               "cfg": {"factor_sumidero_C": 0.0}},
            # Limpieza: la misma fuente 30 d (T_fuente = 30 d) y después apagada,
            # 120 d en total; FEniCS guarda el balance de soluto por paso.
            **{f"S{2 + int(con)}_limpieza_{'con' if con else 'sin'}": {
                "desc": f"Fuente 30 d y 90 d de limpieza, factor_sumidero_C = {int(con)}",
                "cfg": {"factor_sumidero_C": float(con), "t_fuente_off": 30 * 86400,
                        "T_fuente": 30 * 86400, "T": 120 * 86400, "balance_masa": True},
            } for con in (True, False)},
        },
    },

    # ── N. Modelo sin W- con almacenamiento (docs/adjunto_sin_sumidero.tex) ──
    # Ecuación de transporte de la forma no conservativa: sin el sumidero
    # explícito (factor_sumidero_C = 0) y con -S_s ∂ₜh·C (factor_almacenamiento_C = 1).
    # Las 6 variantes: {adr, mrmt_semi, mrmt_block} × {BFR, FEniCS}.
    "N_modelo_nuevo": {
        "desc": "ADR / MRMT Semi / MRMT Bloque sin W- y con almacenamiento, BFR y FEniCS",
        "pregunta": "¿Las 6 variantes del modelo sin W- corren y concuerdan entre backends?",
        "casos": {
            f"N_{m}": {"desc": f"{m}, sin W-, con -S_s ∂ₜh·C",
                       "cfg": {"model": m, "extract_C": False, "factor_sumidero_C": 0.0,
                               "factor_almacenamiento_C": 1.0}}
            for m in ("adr", "mrmt_semi", "mrmt_block")
        },
    },

    # ── P. Presentación: 6 variantes con adjunto (videos de grilla 3×2) ─────
    # Sustituye a scripts/generar_comparacion.py y run_all_6_variants.py.
    # Los casos se llaman como el modelo para que EXP_ROOT funcione con
    # scripts/animate_*.py (<EXP_ROOT>/<modelo>/<backend>/).
    "P_presentacion": {
        "desc": "ADR / MRMT Semi / MRMT Bloque × BFR / FEniCS, forward + adjunto",
        "comparar": False,
        "casos": {
            m: {"desc": f"{m}, run_type=comparison", "cfg": {"model": m, "run_type": "comparison"}}
            for m in ("adr", "mrmt_semi", "mrmt_block")
        },
    },
}


def iter_casos(grupos=None, casos=None):
    """Genera (grupo, caso, spec) filtrando por nombre de grupo y/o caso."""
    for g, gspec in GRUPOS.items():
        if grupos and g not in grupos:
            continue
        for c, spec in gspec["casos"].items():
            if casos and c not in casos and f"{g}/{c}" not in casos:
                continue
            yield g, c, spec


def config_caso(base, grupo, caso, spec, backend, out_root):
    """Config completa (base + cambios) de un caso para un backend."""
    import os
    cfg = dict(base)
    cfg.update(spec.get("cfg", {}))
    cfg.update(spec.get(backend, {}))
    cfg["metodo"] = backend
    cfg["exp_dir"] = os.path.join(out_root, grupo, caso)
    return cfg
