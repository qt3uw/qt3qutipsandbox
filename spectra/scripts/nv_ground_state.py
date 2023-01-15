from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import

from qutip import jmat, tensor, identity, Qobj

uB = None
uN = None


@dataclass
class NVGroundParameters14N:
    nuclear_spin: int = 1
    electron_spin: int = 1
    f_fine_structure: float = 2.88E9
    f_nuclear_quadrupole: float = -5.01E6
    f_axial_magnetic_hyperfine: float = -2.14E6
    f_transverse_magnetic_hyperfine: float = -2.7E6
    g_factor_electron: float = 2.0028

def nnplus1(n):
    return n * (n + 1)

def twonplus1(n):
    return 2 * n + 1
def get_nv_ground_hamiltonian(p:NVGroundParameters14N) -> Qobj:
    """
    Gets ground state hamiltonian in frequency units (h=1) with representation ordering S, I
    :param p:
    :return:
    """
    hh = p.f_fine_structure * tensor(jmat(p.electron_spin, 'z') ** 2 - (nnplus1(p.electron_spin) / 3) *
                                                                        identity(twonplus1(p.electron_spin)),
                                     identity(twonplus1(p.nuclear_spin))) + \
         p.f_axial_magnetic_hyperfine * tensor(jmat(p.electron_spin, 'z'), jmat(p.nuclear_spin, 'z')) + \
         p.f_transverse_magnetic_hyperfine * (tensor(jmat(p.electron_spin, 'x'), jmat(p.nuclear_spin, 'x')) +
                                              tensor(jmat(p.electron_spin, 'y'), jmat(p.nuclear_spin, 'y'))) + \
         p.f_nuclear_quadrupole * tensor(identity(twonplus1(p.electron_spin)), (jmat(p.nuclear_spin, 'z') ** 2 -
                                                                                identity(twonplus1(p.nuclear_spin)))* \
                                                                                (nnplus1(p.nuclear_spin) / 3))
    return hh

def get_nv_interaction_hamiltonian:
    raise(NotImplementedError)



def get_nv_ground_eigenspectrum(p: NVGroundParameters14N):
    hh = get_nv_ground_hamiltonian(p)
    #print(hh.eigenstates())
    energies = hh.eigenenergies()
    energies = (energies - np.min(energies)) * 1.E-6
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
    ax1.set_ylabel('energy (MHz)')
    top_e_span = energies[-1] - energies[3]
    bottom_e_span = energies[2] - energies[0]
    ax0.set_ylim(energies[3] - 0.05 * top_e_span, energies[-1] + 0.05 * top_e_span)
    ax1.set_ylim(energies[0] - 0.05 * bottom_e_span, energies[2] + 0.05 * bottom_e_span)
    ax0.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax0.xaxis.tick_top()
    ax0.tick_params(labeltop=False)  # don't put tick labels at the top
    ax1.xaxis.tick_bottom()
    ax1.set_xticks([])
    for a in [ax0, ax1]:
        a.hlines(energies, color='k', xmin=0, xmax=1)
        a.grid()
    plt.show()


if __name__ == "__main__":
    get_nv_ground_eigenspectrum(NVGroundParameters14N())