from dataclasses import dataclass
from typing import List, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants, h

from qutip import jmat, tensor, identity, Qobj

uB = physical_constants['Bohr magneton'][0]
uN = physical_constants['nuclear magneton'][0]


@dataclass
class NVGroundParameters14N:
    nuclear_spin: int = 1
    electron_spin: int = 1
    f_fine_structure: float = 2.88E9
    f_nuclear_quadrupole: float = -5.01E6
    f_axial_magnetic_hyperfine: float = -2.14E6
    f_transverse_magnetic_hyperfine: float = -2.7E6
    g_factor_electron: float = 2.0028
    gyromagnetic_constant_nuclear: float = 1.93297E7 / (2 * np.pi) # Hz / Tesla


def nnplus1(n):
    return n * (n + 1)


def twonplus1(n):
    return 2 * n + 1


def get_nv_ground_hamiltonian(p:NVGroundParameters14N) -> Qobj:
    """
    Gets ground state hamiltonian in frequency units (h=1) with representation ordering S, I
    From Doherty et a., Physics Reports 528 (2013)
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


def get_nv_zeeman_hamiltonian(p:NVGroundParameters14N, bvector):
    h_zeeman = uB / h * p.g_factor_electron * tensor(jmat(p.electron_spin, 'x') * bvector[0] +
                                            jmat(p.electron_spin, 'y') * bvector[1] +
                                            jmat(p.electron_spin, 'z') * bvector[2], identity(twonplus1(p.nuclear_spin))) + \
          p.gyromagnetic_constant_nuclear * (tensor(identity(twonplus1(p.electron_spin)),
                                                    jmat(p.nuclear_spin, 'x') * bvector[0] +
                                                    jmat(p.nuclear_spin, 'y') * bvector[1] +
                                                    jmat(p.nuclear_spin, 'z') * bvector[2]))
    return h_zeeman


def get_nv_ground_eigenspectrum(p: NVGroundParameters14N, bvector = np.zeros(3)):
    hh = get_nv_ground_hamiltonian(p) + get_nv_zeeman_hamiltonian(p, bvector)
    energies = hh.eigenenergies()
    eigenstates = hh.eigenstates()
    return energies, eigenstates


def get_nv_ground_transition_matrix_elements(transition_hamiltonian):
    #TODO
    raise(NotImplementedError)

def get_eigenstate_amplitude_string(eigenstate: Qobj, eigenstate_labels: List[str]):
    res = ''
    for i, l in enumerate(eigenstate_labels):
        res += f'{eigenstate_labels[i]}: {np.abs(eigenstate.data) ** 2}\n\r'
    return res


def plot_eigenspectrum(energies: List[float], eigenstates: List[Qobj], eigenstate_labels: List[str]=None, y_label=None):
    """
    Plots energies as a function of eigenstate, with labels that give the probabilities in the uncoupled basis
    :param energies:
    :param eigenstates:
    :param eigenstate_labels:
    :param y_label:
    :return:
    """
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    fig = go.Figure()
    import plotly.express as px
    import numpy as np

    color = px.colors.sequential.Plasma[0]
    raise(NotImplementedError('stopped here: eigenstate data format does not make sense to me yet'))
    print(get_eigenstate_amplitude_string(eigenstates[0], eigenstate_labels))
    bar = go.Bar(
        y=energies, name='foo',
        marker_color=color
    )

    bar.showlegend = False
    #
    # fig.update_yaxes(range=[cut_interval[1], max(df.max() * 1.1)], row=1, col=1)
    # fig.update_xaxes(visible=False, row=1, col=1)
    # fig.update_yaxes(range=[0, cut_interval[0]], row=2, col=1)
    fig.show()

def plot_nv_ground_eigenspectrum(p: NVGroundParameters14N, bvector=np.zeros(3)):
    energies, eigenstates = get_nv_ground_eigenspectrum(p, bvector=bvector)
    energies = (energies - np.min(energies)) * 1.E-6
    eigenstate_labels = ['|-1>|-1>', ] * 9 #FIXME
    plot_eigenspectrum(energies, eigenstates, eigenstate_labels, y_label='Energy (MHz)')
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


def plot_allowed_transitions(p: NVGroundParameters14N, hamiltonian):
    #TODO
    raise(NotImplementedError)

if __name__ == "__main__":
    plot_nv_ground_eigenspectrum(NVGroundParameters14N(), bvector=np.array([0., 0., 300.E-4]))