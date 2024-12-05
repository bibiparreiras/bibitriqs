from triqs.gf import *
import numpy as np
from math import pi

# Definição do solver IPT
class IPTSolver:
    def __init__(self, beta):
        self.beta = beta

        # Matsubara frequency Green's functions
        iw_mesh = MeshImFreq(beta=beta, S='Fermion', n_iw=1001)
        self.G_iw = Gf(mesh=iw_mesh, target_shape=[1, 1])
        self.G0_iw = self.G_iw.copy()  # self.G0 will be set by the user after initialization
        self.Sigma_iw = self.G_iw.copy()

        # Imaginary time
        tau_mesh = MeshImTime(beta=beta, S='Fermion', n_tau=10001)
        self.G0_tau = Gf(mesh=tau_mesh, target_shape=[1, 1])
        self.Sigma_tau = self.G0_tau.copy()

    def solve(self, U):
        self.G0_tau << Fourier(self.G0_iw)
        self.Sigma_tau << (U**2) * self.G0_tau * self.G0_tau * self.G0_tau
        self.Sigma_iw << Fourier(self.Sigma_tau)

        # Dyson
        self.G_iw << inverse(inverse(self.G0_iw) - self.Sigma_iw)

from triqs.plot.mpl_interface import *
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100

# Primeira parte: Loop iterativo com t, U e n_loops
t = 1.0
U = 5.0
beta = 20
n_loops = 25

S = IPTSolver(beta=beta)
S.G_iw << SemiCircular(2 * t)

fig1 = plt.figure(figsize=(12, 8))

for i in range(n_loops):
    S.G0_iw << inverse(iOmega_n - t**2 * S.G_iw)
    S.solve(U=U)
    #print('iteração = ', i)
    # Real axis function with Pade approximation
    G_w = Gf(mesh=MeshReFreq(window=(-5.0, 5.0), n_w=1000), target_shape=[1, 1])
    G_w.set_from_pade(S.G_iw, 100, 0.01) 
    G0_w = Gf(mesh=MeshReFreq(window=(-5.0, 5.0), n_w=1000), target_shape=[1, 1])
    G0_w.set_from_pade(S.G0_iw, 100, 0.01)
    #sigma = 
    #print(G_w.data[0,0,0])
    #print(G_w.data.shape)
    if i % 8 == 0:
        oplot(-G_w.imag / pi, figure=fig1, label=f"Iteration = {i+1}", name=r"$\rho$")
#for j in range(0, 999):
#    print(G_w.mesh,G_w.data.real[j,0,0],G_w.data.imag[j,0,0])

for j, mesh_point in enumerate(G_w.mesh): 
    omega = mesh_point.value  # Extrai o valor da frequência (ω)
    real_part_G = G_w.data.real[j, 0, 0]  # Parte real
    imag_part_G = G_w.data.imag[j, 0, 0]  # Parte imaginária

    # Para Σ(ω)
    real_part_Sigma = S.Sigma_iw.data.real[j, 0, 0]  # Parte real de Σ(ω)
    imag_part_Sigma = S.Sigma_iw.data.imag[j, 0, 0]  # Parte imaginária de Σ(ω)

    # Imprime as partes de G(ω) e Σ(ω)
    print(f"Frequência (ω): {omega:.3f}, Re[G(ω)]: {real_part_G:.6f}, Im[G(ω)]: {imag_part_G:.6f}, "
          f"Re[Σ(ω)]: {real_part_Sigma:.6f}, Im[Σ(ω)]: {imag_part_Sigma:.6f}")


plt.legend()
plt.savefig("grafico1.png", dpi=300)  # Salva o primeiro gráfico

# Segunda parte: Variação de U
fig2 = plt.figure(figsize=(6, 6))
pn = 0  # iteration counter for plotting

for U in np.arange(1.0, 10.0, 1.0):

    S = IPTSolver(beta=beta)
    S.G_iw << SemiCircular(2 * t)

    # DMFT
    for i in range(n_loops):
        S.G0_iw << inverse(iOmega_n - t**2 * S.G_iw)
        S.solve(U)

    # Real-axis with Pade approximation
    G_w = Gf(mesh=MeshReFreq(window=(-8.0, 8.0), n_w=1000), target_shape=[1, 1])
    G_w.set_from_pade(S.G_iw, 100, 0.01)

    # Plotting
    ax = fig2.add_axes([0, 1.0 - (pn + 1) / 6.0, 1, 1.0 / 6.0])  # subplot
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    oplot(-G_w.imag / pi, linewidth=3, label=f"U = {U:.2f}")
    plt.xlim(-8, 8)
    plt.ylim(0, 0.35)
    plt.ylabel("")
    pn = pn + 1

plt.savefig("grafico2.png", dpi=300)  # Salva o segundo gráfico