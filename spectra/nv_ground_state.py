from dataclasses import dataclass
from typing import List, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

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


def get_eigenstate_amplitude_hovertext(eigenstate: Qobj, eigenstate_labels: List[str]):
    res = '|Jz>|Iz>: probability<br>'
    probs = np.abs(eigenstate.full().flatten()) ** 2
    for i, l in enumerate(eigenstate_labels):
        res += f'{eigenstate_labels[i]}: {probs[i]:.4f}<br>'
    return res

def get_transition_amplitudes(transition_operator: Qobj, energies: Sequence[float], eigenstates: Sequence[Qobj]):
    """
    Given a transition operator, set of energies and energy eigenstates computes the transition energies and amplitudes
    :param transition_operator: Operator of the transition matrix written in the energy eigenbasis.
    :param energies: Energies
    :param eigenstates: Energy eigenstates corresponding to energies
    :return: transition energies, transition amplitudes
    """
    transition_energies = []
    transition_amplitudes = []
    for i, psi_i in enumerate(eigenstates):
        for j, psi_j in enumerate(eigenstates):
            transition_energy = energies[j] - energies[i]
            if transition_energy > 0:
                transition_energies.append(transition_energy)
                transition_amplitudes.append(np.abs(transition_operator.matrix_element(psi_j.conj(), psi_i)))
    transition_energies = np.array(transition_energies)
    transition_amplitudes = np.array(transition_amplitudes)
    return transition_energies, transition_amplitudes

def get_magnetic_transition_operator(p:NVGroundParameters14N, transition_bvec) -> Qobj:
    """
    Gets the zeeman transition hamiltonian
    :param p: State parameters for an NV- like state
    :param transition_bvec: pk-pk magnetic field vector for oscillating field
    :return: transition operator Qobj
    """
    jjs = jmat(p.electron_spin)
    iis = jmat(p.nuclear_spin)
    electron_moment = uB * p.g_factor_electron * transition_bvec / h
    nuclear_moment = p.gyromagnetic_constant_nuclear * transition_bvec
    hh_int = tensor((electron_moment[0] * jjs[0] + electron_moment[1] * jjs[1] + electron_moment[2] * jjs[2]),
                    identity(twonplus1(p.electron_spin))) + \
             tensor(identity(twonplus1(p.electron_spin)),
                    (nuclear_moment[0] * iis[0] + nuclear_moment[1] * iis[1] + nuclear_moment[2] * iis[2]))
    return hh_int


def plot_transition_amplitudes(transition_operator: Qobj, energies: Sequence[float], eigenstates: Sequence[Qobj],
                               fig=None, xscale=1., xlabel=None, yscale=1., ylabel='rabi frequency (Hz)', title=None):
    """
    Plots all of the transition amplitudes
    :param transition_operator:
    :param energies:
    :param eigenstates:
    :param fig:
    :return:
    """
    hovertip_text = []

    for i in range(energies.shape[0]):
        for j in range(energies.shape[0]):
            if (energies[j] - energies[i]) > 0:
                hovertip_text.append(f'|{i}> --> |{j}>')

    transition_energies, transition_amplitudes = get_transition_amplitudes(transition_operator, energies, eigenstates)

    if fig is None:
        fig = go.Figure()

    scatter = go.Scatter(x=transition_energies * xscale, y=transition_amplitudes * yscale, hovertext=hovertip_text,
                         mode='markers')
    fig.add_trace(scatter)
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)


    return fig


def plot_eigenspectrum(energies: Sequence[float], eigenstates: Sequence[Qobj], eigenstate_labels: List[str]=None,
                       ylabel=None, yscale=1., fig:go.Figure=None):
    """
    Plots energies as a function of eigenstate, with labels that give the probabilities in the uncoupled basis
    :param energies:
    :param eigenstates:
    :param eigenstate_labels:
    :param y_label:
    :return:
    """

    energies = (energies - np.min(energies)) * yscale
    color = px.colors.sequential.Plasma[0]
    if fig is None:
        fig = go.Figure()

    state_strings = []
    for state in eigenstates:
        state_strings.append(get_eigenstate_amplitude_hovertext(state, eigenstate_labels))
    scatter = go.Scatter(
        y=energies, hovertext=state_strings,
        marker_color=color, line={'width':0}, mode='markers', marker=dict(size=12,
                                                                          line=dict(width=2, color='DarkSlateGrey'))
    )
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text='state index')
    fig.add_trace(scatter)
    return fig

def plot_nv_ground_eigenspectrum(p: NVGroundParameters14N, bvector=np.zeros(3)):
    energies, eigenstates = get_nv_ground_eigenspectrum(p, bvector=bvector)
    energies = (energies - np.min(energies))
    # FIXME: Do this more automatically
    eigenstate_labels = ['|1>|1>', '|1>|0>', '|1>|-1>', '|0>|1>', '|0>|0>', '|0>|-1>', '|-1>|1>', '|-1>|0>', '|-1>|-1>']
    fig = plot_eigenspectrum(energies, eigenstates, eigenstate_labels, ylabel='Energy (MHz)', yscale=1.E-6)
    fig.update_layout(title=dict(text=f'B = ({bvector[0] * 1.E4:.2f} G, {bvector[1] * 1.E4:.2f} G, '
                                      f'{bvector[2] * 1.E4:.2f} G)'))
    fig.show()

def plot_nv_ground_magnetic_transition_amplitudes(transition_bvec, static_bvec,
                                                  p: NVGroundParameters14N=NVGroundParameters14N()):
    """
    NV axis is in the (0, 0, 1) direction
    This has been checked against https://doi.org/10.1038/s41598-020-61669-w for the case where NV-axis is aligned to field.
    :param transition_bvec: RF field vector in Tesla
    :param static_bvec: Bias field vector in Tesla
    :param p: Parameters
    :return: plotly figure
    """
    p=NVGroundParameters14N()
    hh_int = get_magnetic_transition_operator(p, transition_bvec)
    energies, eigenstates = get_nv_ground_eigenspectrum(p, static_bvec)
    fig = plot_transition_amplitudes(hh_int, energies, eigenstates, xscale=1.E-6, xlabel='transition frequency (MHz)',
                                     yscale=1.E-3, ylabel='rabi frequency (kHz)')
    fig.update_layout(title=dict(text=f'B = ({static_bvec[0] * 1.E4:.2f} G, {static_bvec[1] * 1.E4:.2f} G, '
                                      f'{static_bvec[2] * 1.E4:.2f} G)<br>'
                                      f'B_drive = ({transition_bvec[0] * 1.E4:.2f} G, '
                                      f'{transition_bvec[1] * 1.E4:.2f} G, '
                                      f'{transition_bvec[2] * 1.E4:.2f} G)'))
    fig.show()
    return fig


if __name__ == "__main__":
    bmag = 4.6E-3
    phi = 0. * np.pi / 180. # Polar angle
    theta = 0.
    static_bvec = bmag * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
    transition_bvec = np.array([0., 1.E-4, 0.E-4])

    plot_nv_ground_magnetic_transition_amplitudes(transition_bvec=transition_bvec,
                                                  static_bvec=static_bvec)
    plot_nv_ground_eigenspectrum(NVGroundParameters14N(), bvector=static_bvec)
