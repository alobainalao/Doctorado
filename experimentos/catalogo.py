"""
Catálogo de experimentos BFR (RBF-FD) vs FEniCS (FEM).

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

GRUPOS = {
    # ── A. Términos fuente: aislar la contribución de fuente y pozo ──────────
    "A_terminos": {
        "desc": "Términos fuente: CI sola, sólo fuente, sólo pozo, ambos",
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
        "casos": {
            f"E_dt{dt}": {"desc": f"dt={dt} s ({dt / 3600:g} h)", "cfg": {"dt": dt}}
            for dt in (36000, 7200)
        },
    },

    # ── F. Estabilización del transporte en BFR (upwind) ────────────────────
    "F_upwind_bfr": {
        "desc": "Upwind direccional en el transporte BFR",
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
                "cfg": {"extract_C": con, "t_fuente_off": 2592000, "T": T_W,
                        "dt": 36000, "balance_masa": True},
                "backends": ("fenicsx",),
            }
            for con in (True, False)
        },
    },

    # ── Z. Cuadrado contaminado, sólo pozo (decisión 1, dinámica en el pozo) ─
    # Dominio cuadrado de 1 km, pozo en el centro, condición libre de salida
    # (pfs) en todo el borde, transporte con flujo dispersivo nulo en el borde,
    # C = 1 y u = 0 al inicio, sin fuente. Con C uniforme la forma no
    # conservativa sin W- debe conservar C = 1: el soluto sale con el agua.
    "Z_cuadrado": {
        "desc": "Cuadrado con C=1 inicial y pozo central; con y sin W-",
        "pregunta": "¿Sin W- el soluto sale por el pozo o se acumula?",
        "comparar": False,
        "casos": {
            f"Z{int(con)}_{'con' if con else 'sin'}_sumidero": {
                "desc": f"{'Con' if con else 'Sin'} W-, cuadrado C=1, pfs en todo el borde",
                "cfg": {"domain": "cuadrado", "mesh_pkl": None, "activate_fuente": False,
                        "extract_C": con, "bc_outlet_mode": "dirichlet", "phi_const": 0.1,
                        "C_init_const": 1.0, "h_init_const": 300.0,
                        "lado_cuadrado": 600.0, "C_banda_limpia": 60.0, "T": T_Z,
                        "S_s_const": 1e-5, "Q_std": 0.01,
                        "spacing": 30, "min_sp": 4},
                "backends": ("bfr",),
            }
            for con in (True, False)
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
