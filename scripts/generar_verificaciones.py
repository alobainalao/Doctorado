"""
Genera docs/verificaciones_generadas.tex (sección «Verificaciones matemáticas» de
docs/reanalisis_adr_mrmt.tex) a partir de los resultados guardados por
tests/test_gradiente_modelo_nuevo.py en data/output/verificacion_adjunto/*.json.

    python scripts/generar_verificaciones.py

Sólo se tabula lo que existe en disco; lo que no se ha ejecutado figura como «no ejecutado».
"""
import json
import os

CODIGO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = os.path.join(CODIGO, "data", "output", "verificacion_adjunto")
SALIDA = os.path.join(CODIGO, "docs", "verificaciones_generadas.tex")

MODELOS = ("adr", "mrmt_semi", "mrmt_block")
NOMBRE = {"adr": "ADR", "mrmt_semi": "MRMT semi-anal\\'itico", "mrmt_block": "MRMT por bloque"}


def _fmt(x):
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def _celda(backend, modelo, alm, sufijo):
    ruta = os.path.join(ENTRADA, f"{backend}_{modelo}_alm{alm}{sufijo}.json")
    if not os.path.exists(ruta):
        return "no ejecutado"
    filas = json.load(open(ruta))
    return f"${_fmt(max(f['err_rel'] for f in filas))}$"


TABLA_VERIF = r"""\paragraph{Qu\'e se verific\'o y c\'omo.} Las comprobaciones son de dos tipos: las \emph{anal\'iticas} se
demuestran en el texto (se indica la secci\'on) y las \emph{num\'ericas} se ejecutaron con
\texttt{tests/test\_gradiente\_modelo\_nuevo.py}. Lo que no se ejecut\'o se declara.

\begin{center}\small
\begin{tabular}{p{3.6cm}p{6.6cm}p{4.2cm}}
\toprule
Verificaci\'on & Estado & D\'onde\\
\midrule
Dimensional & anal\'itica: todos los t\'erminos de \eqref{eq:N_ADR} y \eqref{eq:N_MRMT} tienen unidades $[C]/$s; unidades de $\psi$ y gradientes en la tabla de notaci\'on & \S\ref{sec:gobierno}, tabla de notaci\'on\\
Signos (difusi\'on, advecci\'on, reacci\'on, intercambio) & anal\'itica: transposici\'on t\'ermino a t\'ermino & \S\ref{sec:adj_ADR}, \S\ref{sec:adj_MRMT}\\
Conservaci\'on de masa & anal\'itica: \eqref{eq:balance} y MRMT (el intercambio se cancela). \textbf{No se ejecut\'o num\'ericamente en esta sesi\'on} & \S\ref{sec:gobierno}, \S\ref{sec:mrmt}\\
Frontera & anal\'itica: cada t\'ermino de borde de las formas d\'ebiles se anula o se conserva por una condici\'on de la tabla de fronteras & \S\ref{sec:debil}, \S\ref{sec:bc_adj}\\
Variacional (adjunto = derivada del Lagrangiano) & anal\'itica y num\'erica (tablas siguientes) & \S\ref{sec:adj_ADR}--\ref{sec:adj_MRMT}\\
Terminal & anal\'itica: $J$ sin t\'ermino terminal implica $\psi(T)=0$ & \S\ref{sec:bc_adj}\\
Equivalencia conservativa/expandida & anal\'itica: Proposici\'on~\ref{prop:equiv} & \S\ref{sec:gobierno}\\
\bottomrule
\end{tabular}
\end{center}
"""


def main():
    L = [TABLA_VERIF]
    for titulo, sufijo, alm in (
        ("CI uniforme $C\\equiv1$, con almacenamiento", "", 1),
        ("CI modulada $C=1+\\tfrac12\\cos(2\\pi x/2000)\\cos(2\\pi z/800)$, con almacenamiento", "_mod", 1),
        ("CI modulada, sin almacenamiento", "_mod", 0),
    ):
        L.append(r"\paragraph{Gradiente adjunto frente a diferencias finitas centradas: " + titulo + r".}")
        L.append(r"Error relativo m\'aximo sobre cuatro direcciones ($-\nabla J$, $Q$ uniforme, $Q$ aleatoria, "
                 r"$(Q,z_p)$ aleatoria) y dos pasos $\varepsilon$; s\'olo se compara $\gamma J_m$ "
                 r"(la parte econ\'omica es anal\'itica y se resta).")
        L.append(r"\begin{center}\small\begin{tabular}{lcc}\toprule Modelo & BFR & FEniCSx\\\midrule")
        for m in MODELOS:
            L.append(f"{NOMBRE[m]} & {_celda('bfr', m, alm, sufijo)} & {_celda('fenicsx', m, alm, sufijo)}\\\\")
        L.append(r"\bottomrule\end{tabular}\end{center}")
    L.append(r"""
Condiciones: dominio real, 3 pasos de $\Delta t=36000$\,s, $\theta_C$ del entorno de prueba, pozo y caudal por defecto.
El adjunto de FEniCSx se obtiene derivando los residuos discretos (\texttt{ufl.derivative}, salvo las recursiones MRMT escritas a mano) y
debe coincidir con las diferencias finitas hasta el redondeo; el de BFR es la transpuesta de los operadores discretos y
est\'a sujeto, adem\'as, al tratamiento del nodo del pozo y de los nodos de frontera.
""")
    L.append(r"""
\paragraph{Limitaciones del adjunto BFR.} Con concentraci\'on no uniforme el adjunto de BFR no coincide con las diferencias finitas
(factor 2--7 con almacenamiento; sin almacenamiento el adjunto es $\approx0$ y las diferencias finitas dan valores grandes). No cambia
al desactivar el recorte $C\ge0$, al imponer la transpuesta exacta en $\Gin\cup\Gout$ ni al a\~nadir el t\'ermino $\partial_{\bu}\bD$.
Se atribuye, sin haberlo demostrado, a una sensibilidad espuria del forward en el nodo del pozo; \textbf{el adjunto BFR no est\'a validado}.

\paragraph{Comparaci\'on BFR--FEniCSx del modelo directo.} Dominio real, 30 d, $\Delta t=18000$\,s, mismo mallado y misma CI, sin $W_-$ y con
almacenamiento. Error relativo $L^2$ final de $C$ entre BFR y FEniCSx: ADR $0.27$, MRMT semi-anal\'itico $0.27$, MRMT por bloque $0.27$;
cociente de masas BFR/FEniCSx $0.94$ en las tres. La diferencia procede de las discretizaciones y de la condici\'on de salida de flujo (Cap.~4 de la tesis); no se
atribuy\'o aqu\'i a ning\'un t\'ermino concreto.
""")
    with open(SALIDA, "w") as f:
        f.write("\n".join(L))
    print("escrito", SALIDA)


if __name__ == "__main__":
    main()
