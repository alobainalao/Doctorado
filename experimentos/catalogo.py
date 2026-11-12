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
    # ── M. Modelo nuevo (sin sumidero W-, con almacenamiento S_s = φ) con la configuración de main ──
    # Q_std = 0.3 (default.json de main desde el 28-oct). Las corridas con 1e-3 quedaron archivadas en *_Q1e-3.
    "M_modelo_nuevo": {
        "desc": "Modelo nuevo ADR / MRMT semi / MRMT bloque, Q_std = 0.3, 30 días",
        "pregunta": "¿El directo del modelo nuevo es consistente entre BFR y FEniCS en los tres modelos?",
        "casos": {
            f"M{i}_{modelo}": {"desc": f"{modelo} sin W-, con almacenamiento, Q_std = 0.3",
                               "cfg": {"model": modelo, "Q_std": 0.3, "factor_almacenamiento_C": 1.0, "bc_outlet_mode": "libre", **MAIN_FUENTE}}
            for i, modelo in enumerate(("adr", "mrmt_semi", "mrmt_block"))
        },
    },

    # ── MC. Modelo nuevo en la malla con conos (preprocessing/malla_conos.py): misma configuración que M ──
    "MC_malla_conos": {
        "desc": "Modelo nuevo ADR / MRMT semi / MRMT bloque en la malla con conos, Q_std = 0.3, 30 días",
        "pregunta": "¿Cambia el directo al refinar la pluma, la columna del pozo y la zona de la optimización?",
        "casos": {
            f"MC{i}_{modelo}": {"desc": f"{modelo} como M{i}_{modelo}, malla data/input/malla_conos/malla.pkl",
                                "cfg": {"model": modelo, "Q_std": 0.3, "factor_almacenamiento_C": 1.0, "bc_outlet_mode": "libre",
                                        **MAIN_FUENTE, "mesh_pkl": "data/input/malla_conos/malla.pkl"}}
            for i, modelo in enumerate(("adr", "mrmt_semi", "mrmt_block"))
        },
    },
    "M_adjunto": {
        "desc": "ADR / MRMT Semi / MRMT Bloque, configuración de main + fuente común, salida libre; forward + adjunto",
        "pregunta": "¿Cómo son los adjuntos ψ_h, ψ_C del modelo nuevo con la configuración final?",
        "comparar": False,
        "casos": {
            f"MA_{m}": {"desc": f"{m}, modelo nuevo, salida libre, run_type=comparison",
                        "cfg": {"model": m, "run_type": "comparison", "extract_C": False, "Q_std": 0.3,
                                "factor_almacenamiento_C": 1.0,
                                "bc_outlet_mode": "libre", **MAIN_FUENTE}}
            for m in ("adr", "mrmt_semi", "mrmt_block")
        },
    },
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
