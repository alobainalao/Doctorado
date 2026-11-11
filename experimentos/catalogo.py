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
# Grupo M: configuración de main.py (default.json + malla_sp60: dt 18000 s, 30 d, n_stencil 25, Q_std 0.3) salvo la malla compartida
# (necesaria para que BFR y FEniCSx usen los mismos nodos). La fuente es más estrecha en z y menos intensa, igual en todas las variantes.
MAIN_FUENTE = {"n_stencil": 25, "epsilon_z_fuente": 30.0, "intensidad_fuente": 5e-12}

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

    # ── H. Ablación del flujo BFR↔FEM: de la configuración antigua a la actual ──────
    # Pregunta: ¿por qué H y |U| diferían tanto entre BFR y FEM en main y ahora casi no?
    # Cada caso suma UN cambio al anterior (acumulativo). Horizonte corto (12 pasos): sólo
    # importa el flujo. Los cambios son los de config/default.json del 2026-10-24/28 y la malla saneada.
    "H_ablacion_flujo": {
        "desc": "Ablación acumulativa: configuración antigua → actual (sólo flujo, 12 pasos)",
        "pregunta": "¿Qué cambio explica que H y |U| de BFR y FEM ahora casi coincidan?",
        "casos": {
            nombre: {"desc": desc,
                     "cfg": {**SIN_FUENTE, "T": 12 * 18000, **ac},
                     **({"bfr": extra_bfr, "backends": ["bfr"], "ref_fem": "H_ablacion_flujo/H5_theta1"} if extra_bfr else {})}
            for nombre, desc, ac, extra_bfr in (
                ("H0_antiguo", "Config. antigua: malla sin sanear, Q=1e-3, kernel 2.5·h por suma, θ=0.9",
                 {"mesh_pkl": "data/input/malla_compartida/malla_2026-10-22_sin_saneo.pkl", "r_pozo": None,
                  "norm_delta_pozo": "suma", "Q_std": 1e-3, "theta": 0.9, "theta_C": 0.9}, None),
                ("H1_Q0p3", "H0 con Q_std = 0.3",
                 {"mesh_pkl": "data/input/malla_compartida/malla_2026-10-22_sin_saneo.pkl", "r_pozo": None,
                  "norm_delta_pozo": "suma", "Q_std": 0.3, "theta": 0.9, "theta_C": 0.9}, None),
                ("H2_malla_saneada", "H1 con la malla saneada",
                 {"r_pozo": None, "norm_delta_pozo": "suma", "Q_std": 0.3, "theta": 0.9, "theta_C": 0.9}, None),
                ("H3_r_pozo20", "H2 con r_pozo = 20 m (soporte fijo del kernel)",
                 {"norm_delta_pozo": "suma", "Q_std": 0.3, "theta": 0.9, "theta_C": 0.9}, None),
                ("H4_delta_area", "H3 con delta normalizada por área",
                 {"Q_std": 0.3, "theta": 0.9, "theta_C": 0.9}, None),
                ("H5_theta1", "H4 con θ = θ_C = 1 (configuración actual)", {}, None),
                ("H6_bc_fem", "H5 con bc_outlet_mode = fem (sólo BFR)", {}, {"bc_outlet_mode": "fem"}),
            )
        },
    },

    # ── M. Modelo nuevo (sin sumidero W-, con almacenamiento S_s = φ) con la configuración de main ──
    # Q_std = 0.3 (default.json de main desde el 28-oct). Las corridas con 1e-3 quedaron archivadas en *_Q1e-3.
    "M_modelo_nuevo": {
        "desc": "Modelo nuevo ADR / MRMT semi / MRMT bloque, Q_std = 0.3, 30 días",
        "pregunta": "¿El directo del modelo nuevo es consistente entre BFR y FEniCS en los tres modelos?",
        "casos": {
            f"M{i}_{modelo}": {"desc": f"{modelo} sin W-, con almacenamiento, Q_std = 0.3",
                               "cfg": {"model": modelo, "Q_std": 0.3, "factor_sumidero_C": 0.0,
                                       "factor_almacenamiento_C": 1.0, "bc_outlet_mode": "libre", **MAIN_FUENTE}}
            for i, modelo in enumerate(("adr", "mrmt_semi", "mrmt_block"))
        } | {
            "M3_adr_bc_fem": {"desc": "ADR con el cierre (B) Neumann u·n=f_out en ambos backends (bc_outlet_mode = fem): referencia del cierre anterior",
                              "cfg": {"model": "adr", "Q_std": 0.3, "factor_sumidero_C": 0.0,
                                      "factor_almacenamiento_C": 1.0, "bc_outlet_mode": "fem", **MAIN_FUENTE}},
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

    # ── N_adjunto: las 6 variantes del modelo nuevo con forward + adjunto (run_type=comparison) ──
    # Misma configuración física que N_modelo_nuevo; sólo cambia run_type para que se guarde adjoint_results.npz.
    "N_adjunto": {
        "desc": "ADR / MRMT Semi / MRMT Bloque sin W- y con almacenamiento, forward + adjunto",
        "pregunta": "¿Cómo son los adjuntos ψ_h, ψ_C del modelo nuevo en BFR y FEniCS?",
        "comparar": False,
        "casos": {
            f"NA_{m}": {"desc": f"{m}, sin W-, con -S_s ∂ₜh·C, run_type=comparison",
                        "cfg": {"model": m, "run_type": "comparison", "extract_C": False,
                                "factor_sumidero_C": 0.0, "factor_almacenamiento_C": 1.0}}
            for m in ("adr", "mrmt_semi", "mrmt_block")
        },
    },

    # ── M_adjunto: forward + adjunto del modelo nuevo con la configuración de main (grupo M) y salida libre ──
    "M_adjunto": {
        "desc": "ADR / MRMT Semi / MRMT Bloque, configuración de main + fuente común, salida libre; forward + adjunto",
        "pregunta": "¿Cómo son los adjuntos ψ_h, ψ_C del modelo nuevo con la configuración final?",
        "comparar": False,
        "casos": {
            f"MA_{m}": {"desc": f"{m}, modelo nuevo, salida libre, run_type=comparison",
                        "cfg": {"model": m, "run_type": "comparison", "extract_C": False, "Q_std": 0.3,
                                "factor_sumidero_C": 0.0, "factor_almacenamiento_C": 1.0,
                                "bc_outlet_mode": "libre", **MAIN_FUENTE}}
            for m in ("adr", "mrmt_semi", "mrmt_block")
        },
    },

    # ── N_fuente: como N_adjunto pero con la fuente de contaminante más estrecha en z y menos intensa ──
    # ε_z de la fuente baja de 60 m (spacing·sigma_fuente) a 30 m y su intensidad de 1e-11 a 5e-12 (valores en
    # la config); el resto es idéntico a N_adjunto.
    "N_fuente": {
        "desc": "N_adjunto con ε_z de la fuente = 30 m e intensidad = 5e-12",
        "pregunta": "¿Cómo cambian la pluma y los adjuntos con una fuente más estrecha en z y menos intensa?",
        "comparar": False,
        "casos": {
            f"NA_{m}": {"desc": f"{m}, sin W-, con -S_s ∂ₜh·C, fuente ε_z=30 m, intensidad=5e-12",
                        "cfg": {"model": m, "run_type": "comparison", "extract_C": False,
                                "factor_sumidero_C": 0.0, "factor_almacenamiento_C": 1.0,
                                "epsilon_z_fuente": 30.0, "intensidad_fuente": 5e-12}}
            for m in ("adr", "mrmt_semi", "mrmt_block")
        },
    },

    # ── N_fuente_Q: N_fuente con el caudal del pozo Q_std = 1e-3 (valor de default.json hasta el 28-oct) ──
    "N_fuente_Q": {
        "desc": "N_fuente con Q_std = 1e-3",
        "pregunta": "¿Cómo cambian el cono de h, la pluma y los adjuntos con Q_std = 1e-3 en vez de 0.3?",
        "comparar": False,
        "casos": {
            f"NA_{m}": {"desc": f"{m}, sin W-, con -S_s ∂ₜh·C, fuente ε_z=30 m, intensidad=5e-12, Q_std=1e-3",
                        "cfg": {"model": m, "run_type": "comparison", "extract_C": False,
                                "factor_sumidero_C": 0.0, "factor_almacenamiento_C": 1.0,
                                "epsilon_z_fuente": 30.0, "intensidad_fuente": 5e-12, "Q_std": 1e-3}}
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
