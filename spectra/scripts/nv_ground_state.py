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
    energies, eigenstates = hh.eigenstates()
    return energies, eigenstates


def get_nv_ground_transition_matrix_elements(transition_hamiltonian):
    #TODO
    raise(NotImplementedError)

def get_eigenstate_amplitude_hovertext(eigenstate: Qobj, eigenstate_labels: List[str]):
    res = '|Jz>|Iz>: probability<br>'
    probs = np.abs(eigenstate.full().flatten()) ** 2
    for i, l in enumerate(eigenstate_labels):
        res += f'{eigenstate_labels[i]}: {probs[i]:.4f}<br>'
    return res


def plot_eigenspectrum(energies: Sequence[float], eigenstates: Sequence[Qobj], eigenstate_labels: List[str]=None,
                       ylabel=None, yscale=1.):
    """
    Plots energies as a function of eigenstate, with labels that give the probabilities in the uncoupled basis
    :param energies:
    :param eigenstates:
    :param eigenstate_labels:
    :param y_label:
    :return:
    """
    import plotly.graph_objs as go
    fig = go.Figure()
    import plotly.express as px

    energies = (energies - np.min(energies)) * yscale
    color = px.colors.sequential.Plasma[0]
    state_strings = []
    for state in eigenstates:
        state_strings.append(get_eigenstate_amplitude_hovertext(state, eigenstate_labels))
    bar = go.Scatter(
        y=energies, hovertext=state_strings,
        marker_color=color, line={'width':0}
    )
    fig.update_yaxes(title_text='Energy (MHz)')

    fig.add_trace(bar)

    bar.showlegend = False
    #
    # fig.update_yaxes(range=[cut_interval[1], max(df.max() * 1.1)], row=1, col=1)
    # fig.update_xaxes(visible=False, row=1, col=1)
    # fig.update_yaxes(range=[0, cut_interval[0]], row=2, col=1)
    fig.show()

def plot_nv_ground_eigenspectrum(p: NVGroundParameters14N, bvector=np.zeros(3)):
    energies, eigenstates = get_nv_ground_eigenspectrum(p, bvector=bvector)
    energies = (energies - np.min(energies))
    # FIXME: Do this more automatically
    eigenstate_labels = ['|1>|1>', '|1>|0>', '|1>|-1>', '|0>|1>', '|0>|0>', '|0>|-1>', '|-1>|1>', '|-1>|0>', '|-1>|-1>']
    plot_eigenspectrum(energies, eigenstates, eigenstate_labels, ylabel='Energy (MHz)', yscale=1.E-6)


def plot_allowed_transitions(p: NVGroundParameters14N, hamiltonian):
    #TODO
    raise(NotImplementedError)

if __name__ == "__main__":
    plot_nv_ground_eigenspectrum(NVGroundParameters14N(), bvector=np.array([0., 0., 0.E-4]))