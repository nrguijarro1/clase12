"""
===============================================================
 MODELO DE DINÁMICA DE DEMANDA EMPRESARIAL
 Ecuación Diferencial de Segundo Orden
---------------------------------------------------------------
 Empresa simulada: "TechRetail S.A."
 Variable modelada: Desviación de la demanda respecto al
                    equilibrio histórico D(t)

 Modelo:  D''(t) + 2ζω₀ D'(t) + ω₀² D(t) = A·cos(ωf·t)

 Donde:
   D(t)   : Desviación de demanda (miles de unidades)
   ζ      : Coeficiente de amortiguamiento del mercado
   ω₀     : Frecuencia natural del sistema (1/trimestre)
   A      : Amplitud de la demanda estacional forzada
   ωf     : Frecuencia de la fuerza estacional
===============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.linalg import eig

# ─────────────────────────────────────────────
# 1.  PARÁMETROS DEL MODELO
# ─────────────────────────────────────────────
omega0 = 1.0        # Frecuencia natural (rad / trimestre)
A      = 3.0        # Amplitud estacional (miles de unidades)
omega_f = 0.6       # Frecuencia de fuerza estacional

# Condiciones iniciales: D(0)=2 (desviación inicial), D'(0)=0 (sin velocidad)
D0     = 2.0
dD0    = 0.0

# Horizonte temporal: 20 trimestres
t_span = (0, 20)
t_eval = np.linspace(0, 20, 1000)


# ─────────────────────────────────────────────
# 2.  FUNCIÓN DEL SISTEMA (forma de estado)
#     x1 = D,  x2 = D'
#     x1' = x2
#     x2' = -omega0² x1 - 2ζω₀ x2 + A·cos(ωf·t)
# ─────────────────────────────────────────────
def sistema(t, y, zeta):
    x1, x2 = y
    dx1 = x2
    dx2 = -omega0**2 * x1 - 2*zeta*omega0 * x2 + A * np.cos(omega_f * t)
    return [dx1, dx2]


# ─────────────────────────────────────────────
# 3.  TRES ESCENARIOS DE AMORTIGUAMIENTO
# ─────────────────────────────────────────────
escenarios = {
    "Subamortiguado\n(ζ = 0.1)  –  mercado inestable":   0.1,
    "Críticamente amortiguado\n(ζ = 1.0)  –  mercado óptimo": 1.0,
    "Sobreamortiguado\n(ζ = 2.5)  –  mercado rígido":    2.5,
}

soluciones = {}
for nombre, zeta in escenarios.items():
    sol = solve_ivp(
        fun=lambda t, y: sistema(t, y, zeta),
        t_span=t_span,
        y0=[D0, dD0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8, atol=1e-10
    )
    soluciones[nombre] = sol


# ─────────────────────────────────────────────
# 4.  SOLUCIÓN HOMOGÉNEA  (A = 0)
#     Para ζ = 0.3 (subamortiguado moderado)
# ─────────────────────────────────────────────
zeta_hom = 0.3
sol_hom = solve_ivp(
    fun=lambda t, y: [-omega0**2 * y[0] - 2*zeta_hom*omega0 * y[1] + 0,
                       y[0]] [::-1],   # reutilizamos la misma función con A=0
    t_span=t_span,
    y0=[D0, dD0],
    t_eval=t_eval,
    method="RK45"
)
# Forma directa más clara:
def sistema_hom(t, y):
    x1, x2 = y
    return [x2, -omega0**2 * x1 - 2*zeta_hom*omega0 * x2]

sol_hom = solve_ivp(sistema_hom, t_span, [D0, dD0], t_eval=t_eval,
                    method="RK45", rtol=1e-8, atol=1e-10)


# ─────────────────────────────────────────────
# 5.  ANÁLISIS DE VALORES PROPIOS
# ─────────────────────────────────────────────
print("=" * 55)
print("  ANÁLISIS DE VALORES PROPIOS (raíces características)")
print("=" * 55)
for nombre, zeta in escenarios.items():
    # Matriz compañera: [0, 1; -omega0², -2ζω₀]
    M = np.array([[0, 1],
                  [-omega0**2, -2*zeta*omega0]])
    eigenvalues = np.linalg.eigvals(M)
    tag = nombre.split('\n')[0]
    print(f"\n  {tag}")
    print(f"    λ₁ = {eigenvalues[0]:.4f}")
    print(f"    λ₂ = {eigenvalues[1]:.4f}")
    disc = (2*zeta*omega0)**2 - 4*omega0**2
    tipo = ("Raíces complejas conjugadas" if disc < 0 else
            "Raíz real doble" if disc == 0 else
            "Raíces reales distintas")
    print(f"    Tipo: {tipo}")

print()


# ─────────────────────────────────────────────
# 6.  FIGURA 1 – Tres escenarios de demanda
# ─────────────────────────────────────────────
fig1, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
fig1.suptitle(
    "TechRetail S.A. — Dinámica de Demanda\n"
    r"$D''(t) + 2\zeta\omega_0\,D'(t) + \omega_0^2\,D(t) = A\cos(\omega_f t)$",
    fontsize=13, fontweight="bold"
)

colores = ["#e74c3c", "#27ae60", "#2980b9"]
for ax, (nombre, sol), color in zip(axes, soluciones.items(), colores):
    ax.plot(t_eval, sol.y[0], color=color, linewidth=2.2, label="D(t) — Demanda")
    ax.plot(t_eval, sol.y[1], color=color, linewidth=1.4,
            linestyle="--", alpha=0.7, label="D'(t) — Tasa de cambio")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Desviación\n(miles de unid.)", fontsize=9)
    ax.set_title(nombre, fontsize=10, pad=4)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Tiempo (trimestres)", fontsize=10)
plt.tight_layout()
plt.savefig("fig1_escenarios_demanda.png", dpi=150, bbox_inches="tight")
print("  Figura 1 guardada: fig1_escenarios_demanda.png")


# ─────────────────────────────────────────────
# 7.  FIGURA 2 – Homogénea vs No Homogénea
# ─────────────────────────────────────────────
# No homogénea con ζ = 0.3
sol_nohom = solve_ivp(
    fun=lambda t, y: sistema(t, y, zeta_hom),
    t_span=t_span,
    y0=[D0, dD0],
    t_eval=t_eval,
    method="RK45", rtol=1e-8, atol=1e-10
)

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig2.suptitle(
    "TechRetail S.A. — Comparación Homogénea vs No Homogénea\n"
    r"($\zeta = 0.3$,  $\omega_0 = 1$)",
    fontsize=13, fontweight="bold"
)

ax1.plot(t_eval, sol_hom.y[0], color="#8e44ad", linewidth=2.2)
ax1.set_title("Ecuación Homogénea  [sin forzamiento estacional]", fontsize=10)
ax1.set_ylabel("D(t)  (miles de unid.)", fontsize=9)
ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax1.grid(True, alpha=0.3)

ax2.plot(t_eval, sol_nohom.y[0], color="#e67e22", linewidth=2.2)
fuerza = A * np.cos(omega_f * t_eval)
ax2.fill_between(t_eval, fuerza, alpha=0.2, color="#e67e22",
                 label=r"Fuerza $A\cos(\omega_f t)$")
ax2.set_title("Ecuación No Homogénea  [con forzamiento estacional]", fontsize=10)
ax2.set_ylabel("D(t)  (miles de unid.)", fontsize=9)
ax2.set_xlabel("Tiempo (trimestres)", fontsize=10)
ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig2_hom_vs_nohom.png", dpi=150, bbox_inches="tight")
print("  Figura 2 guardada: fig2_hom_vs_nohom.png")


# ─────────────────────────────────────────────
# 8.  FIGURA 3 – Plano de fases
# ─────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(13, 5))
fig3.suptitle(
    "TechRetail S.A. — Plano de Fases  (D vs D')",
    fontsize=13, fontweight="bold"
)

for ax, (nombre, sol), color in zip(axes3, soluciones.items(), colores):
    ax.plot(sol.y[0], sol.y[1], color=color, linewidth=1.8)
    ax.plot(sol.y[0][0], sol.y[1][0], "o", color="black",
            markersize=6, label="Inicio")
    ax.plot(sol.y[0][-1], sol.y[1][-1], "s", color="red",
            markersize=6, label="Final")
    ax.set_xlabel("D(t)", fontsize=9)
    ax.set_ylabel("D'(t)", fontsize=9)
    ax.set_title(nombre, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)

plt.tight_layout()
plt.savefig("fig3_plano_fases.png", dpi=150, bbox_inches="tight")
print("  Figura 3 guardada: fig3_plano_fases.png")


# ─────────────────────────────────────────────
# 9.  FIGURA 4 – Análisis de resonancia
# ─────────────────────────────────────────────
omega_range = np.linspace(0.01, 3.0, 500)
zeta_vals   = [0.1, 0.3, 0.7, 1.0]

fig4, ax4 = plt.subplots(figsize=(9, 5))
fig4.suptitle(
    "TechRetail S.A. — Curva de Resonancia\n"
    r"Amplitud estacionaria $|H(\omega_f)|$ vs Frecuencia de forzamiento",
    fontsize=12, fontweight="bold"
)

for zeta in zeta_vals:
    # Amplitud de la respuesta estacionaria particular
    num   = A
    denom = np.sqrt((omega0**2 - omega_range**2)**2 +
                    (2*zeta*omega0*omega_range)**2)
    H = num / denom
    ax4.plot(omega_range, H, linewidth=2, label=f"ζ = {zeta}")

ax4.axvline(omega0, color="gray", linestyle="--", linewidth=1,
            label=f"ω₀ = {omega0} (resonancia)")
ax4.set_xlabel("Frecuencia de forzamiento ωf  (rad/trimestre)", fontsize=10)
ax4.set_ylabel("Amplitud estacionaria  (miles de unid.)", fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 20)

plt.tight_layout()
plt.savefig("fig4_resonancia.png", dpi=150, bbox_inches="tight")
print("  Figura 4 guardada: fig4_resonancia.png")


# ─────────────────────────────────────────────
# 10.  RESUMEN NUMÉRICO
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  RESUMEN DE VALORES MÁXIMOS POR ESCENARIO")
print("=" * 55)
for nombre, sol in soluciones.items():
    tag = nombre.split('\n')[0]
    Dmax = np.max(np.abs(sol.y[0]))
    print(f"  {tag:<35}  |D|_max = {Dmax:.3f} mil unid.")

print()
print("  Todos los gráficos han sido guardados exitosamente.")
plt.show()
