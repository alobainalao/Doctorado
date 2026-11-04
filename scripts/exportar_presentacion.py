"""
Exporta a la presentación (presentaciones/avance1-doc-sin-sumidero) las salidas del grupo M_modelo_nuevo:
métricas y series temporales como módulo JS y los videos de comparación BFR | FEniCSx.

    python3 scripts/exportar_presentacion.py [ruta_de_la_presentacion]
"""
import json
import os
import shutil
import sys

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(CODIGO), "presentaciones", "avance1-doc-sin-sumidero")
RAIZ = os.path.join(CODIGO, "data", "output", "experimentos", "M_modelo_nuevo")
CASOS = [("M0_adr", "ADR"), ("M1_mrmt_semi", "MRMT Semi"), ("M2_mrmt_block", "MRMT Bloque")]
PASO = 4   # submuestreo de las series (144 pasos -> 36 puntos)
RES = ["Nt", "dt", "n_nodos", "H_rms_final_m", "H_max_final_m", "H_rango_fem_m", "C_L2_rel_final", "C_Linf_rel_final",
       "V_L2_rel_final", "V_L2_rel_max", "masa_ratio_final", "Hpozo_bfr_final", "Hpozo_fem_final"]
SER = ["t_dias", "H_rms_m", "C_L2_rel", "V_L2_rel", "masa_bfr", "masa_fem", "Cpozo_bfr", "Cpozo_fem", "Hpozo_bfr", "Hpozo_fem", "Cmax_bfr", "Cmax_fem"]


def main():
    salida = []
    for caso, etq in CASOS:
        ruta_m = os.path.join(RAIZ, caso, "comparacion", "metricas.json")
        if not os.path.exists(ruta_m):
            print("falta", caso, "(se omite; volver a ejecutar cuando termine)")
            continue
        cfg = json.load(open(os.path.join(RAIZ, caso, "bfr", "config.json")))
        if cfg.get("bc_outlet_mode") != "libre" or cfg.get("n_stencil") != 25 or cfg.get("epsilon_z_fuente") != 30.0:
            print("omitido", caso, ": sus resultados no son de la configuración final (salida libre, n_stencil 25, fuente 30 m)")
            continue
        if os.path.getmtime(ruta_m) < os.path.getmtime(os.path.join(RAIZ, caso, "bfr", "config.json")):
            print("omitido", caso, ": la comparación es anterior a la última corrida (todavía en curso)")
            continue
        m = json.load(open(ruta_m))
        res = {k: m["resumen"][k] for k in RES}
        ser = {k: [round(float(v), 12) for v in m["series"][k][::PASO]] for k in SER}
        salida.append({"id": caso, "label": etq, "resumen": res, "series": ser})
        dst = os.path.join(PRES, "public", "videos", "sin-sumidero", caso)
        os.makedirs(dst, exist_ok=True)
        for campo in ("H", "V", "C"):
            shutil.copy(os.path.join(RAIZ, caso, "comparacion", f"comparacion_{campo}.mp4"), os.path.join(dst, f"{campo}.mp4"))
    ruta = os.path.join(PRES, "src", "slides-avance3", "modeloNuevo.js")
    with open(ruta, "w") as f:
        f.write("// Generado por codigo-sumidero/scripts/exportar_presentacion.py (grupo M_modelo_nuevo). No editar a mano.\n")
        f.write("export const MODELOS = " + json.dumps(salida, ensure_ascii=False, indent=1) + ";\n")
    print("escrito", ruta)


if __name__ == "__main__":
    main()
