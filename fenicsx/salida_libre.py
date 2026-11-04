"""
Cierre «salida libre» de Γ_out en FEniCSx: derivada normal nula de u, en el sentido de su componente normal
∂ₙ(u·n) = 0 (docs/adjunto_sin_sumidero.tex, §2.2, cierre (D)).

Con Γ_out recta y K_τn = 0, la condición equivale a S_s ∂ₜh − ∂_τ(K_ττ ∂_τ h) = 0 sobre Γ_out: h|Γ_out evoluciona solo, a partir de
h_0|Γ_out, sin depender del interior ni de los controles. El problema de flujo es entonces un problema de Dirichlet con ese dato.
Este módulo evoluciona h|Γ_out con la misma discretización de 3 puntos que BFR (`ecuacion_borde_libre` en funtions/operators.py):
  · nodos de Γ_out sin las dos esquinas (en BFR las esquinas pertenecen al grupo de paredes);
  · K_ττ = K_z (Γ_out vertical), promediada entre vecinos; S_s = φ;
  · θ del flujo; el primer y el último nodo del tramo no tienen vecino a un lado y quedan congelados (∂ₜh = 0).
Devuelve los DOF de h sobre los que imponer Dirichlet y el dato h_Γ en cada paso.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dolfinx import fem


class BordeLibre:
    def __init__(self, G, facet_tags, marker, K_z, S_s, dt, theta, h0):
        mesh = G.mesh
        fdim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(fdim, mesh.topology.dim)
        dofs = np.asarray(fem.locate_dofs_topological(G, fdim, facet_tags.find(marker)))
        z = G.tabulate_dof_coordinates()[dofs, 1]
        orden = np.argsort(z)
        dofs, z = dofs[orden][1:-1], z[orden][1:-1]        # sin las esquinas
        n = len(z)
        Kt, S = np.asarray(K_z)[dofs], np.asarray(S_s)[dofs]
        L = sp.lil_matrix((n, n))
        for i in range(1, n - 1):
            sb, sc = z[i + 1] - z[i], z[i] - z[i - 1]
            kb, kc = 0.5 * (Kt[i] + Kt[i + 1]), 0.5 * (Kt[i] + Kt[i - 1])
            w = 2.0 / (sb + sc)
            L[i, i + 1], L[i, i - 1] = w * kb / sb, w * kc / sc
            L[i, i] = -w * (kb / sb + kc / sc)
        L = L.tocsr()
        M = sp.diags(S / dt)
        self.dofs, self.z = dofs, z
        self._lu = spla.splu((M - theta * L).tocsc())
        self._B = (M + (1.0 - theta) * L).tocsr()
        self._h0 = np.asarray(h0)[dofs].copy()
        self.h = self._h0.copy()

    def reiniciar(self):
        self.h = self._h0.copy()

    def avanzar(self):
        """h_Γ^{n+1} a partir de h_Γ^n (un paso θ)."""
        self.h = self._lu.solve(self._B @ self.h)
        return self.h


def es_libre(p):
    """True si el flujo de FEniCSx debe usar el cierre de salida libre (por defecto en esta rama)."""
    return getattr(p, "bc_outlet_mode", "libre") == "libre"
