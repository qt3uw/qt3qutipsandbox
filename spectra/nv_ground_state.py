from dataclasses import dataclass
from typing import List, Tuple, Sequence
from mpl_toolkits.mplot3d import Axes3D

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
    f_fine_structure: float = 2.87E9
    f_nuclear_quadrupole: float = -5.01E6
    f_axial_magnetic_hyperfine: float = -2.14E6
    f_transverse_magnetic_hyperfine: float = -2.7E6
    g_factor_electron: float = 2.0028
    gyromagnetic_constant_nuclear: float = 1.93297E7 / (2 * np.pi) # Hz / Tesla


def nnplus1(n):
    return n * (n + 1)


def twonplus1(n):
    return 2 * n + 1


def rotate_frame(bvec, nv1, nv2):
    """
    Expresses a lab-frame vector in the nv1 axis system (nv1 is the z-axis)
    :param bvec: vector
    :param nv1: NV orientation
    :param nv2: NV conjugate
    :return: vector in nv1 axis system
    """
    z = nv1
    y = np.cross(nv1, nv2)
    x = np.cross(y, z)

    x = np.array(x)/np.linalg.norm(x)
    y = np.array(y)/np.linalg.norm(y)
    z = np.array(z)/np.linalg.norm(z)

    rot_matrix = np.dot(np.array([x, y, z]), np.array(bvec))
    return rot_matrix


def get_bfields(bvec, nv1, nv2, nv3, nv4):
    """
    Gives magnetic field vectors for all NV configurations. Standard NV-axis configuration is given by [111], [-1-11],
    [-11-1], and [1-1-1].
    :param bvec: magnetic field in lab frame
    :return: an array of magnetic field vectors w.r.t. all four NV-axis oriented coordinate systems.
    """
    b1 = rotate_frame(bvec, nv1, nv2)
    b2 = rotate_frame(bvec, nv2, nv1)
    b3 = rotate_frame(bvec, nv4, nv3)
    b4 = rotate_frame(bvec, nv3, nv4)

    return [b1, b2, b3, b4]


def get_nv_ground_hamiltonian(p:NVGroundParameters14N) -> Qobj:
    """
    Gets ground state hamiltonian in frequency units (h=1) with representation ordering S, I.
    From Doherty et a., Physics Reports 528 (2013).
    :param p: Ground state NV center parameters
    :return: Hamiltonian for interaction between electronic and nuclear spin
    """
    hh = p.f_fine_structure * tensor(jmat(p.electron_spin, 'z') ** 2 - (nnplus1(p.electron_spin) / 3) *
                                                                        identity(twonplus1(p.electron_spin)),
                                     identity(twonplus1(p.nuclear_spin))) + \
         p.f_axial_magnetic_hyperfine * tensor(jmat(p.electron_spin, 'z'), jmat(p.nuclear_spin, 'z')) + \
         p.f_transverse_magnetic_hyperfine * (tensor(jmat(p.electron_spin, 'x'), jmat(p.nuclear_spin, 'x')) +
                                              tensor(jmat(p.electron_spin, 'y'), jmat(p.nuclear_spin, 'y'))) + \
         p.f_nuclear_quadrupole * tensor(identity(twonplus1(p.electron_spin)), (jmat(p.nuclear_spin, 'z') ** 2 -
                                                                                identity(twonplus1(p.nuclear_spin)) *
                                                                                (nnplus1(p.nuclear_spin) / 3)))
    return hh


def get_nv_zeeman_hamiltonian(p:NVGroundParameters14N, bvector):
    """
    Zeeman Hamiltonian for both electronic and nuclear spin.
    :param p: Ground state NV center parameters
    :param bvector: Static magnetic field vector
    :return: The Zeeman Hamiltonian for a static magnetic field
    """
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


def get_transition_amplitudes(transition_operator: Qobj, energies: Sequence[float], eigenstates: Sequence[Qobj],
                              initial_populations: np.ndarray=None):
    """
    Given a transition operator, set of energies and energy eigenstates computes the transition energies and amplitudes
    :param transition_operator: Operator of the transition matrix written in the energy eigenbasis.
    :param energies: Energies
    :param eigenstates: Energy eigenstates corresponding to energies
    :param initial_populations: Initial populations in eigenenergy ordering from ground state
    :return: transition energies, transition amplitudes
    """
    transition_energies = []
    transition_amplitudes = []
    if initial_populations is None:
        initial_populations = np.ones(len(energies))
    initial_populations = initial_populations / np.sum(initial_populations)
    for i, psi_i in enumerate(eigenstates):
        for j, psi_j in enumerate(eigenstates):
            transition_energy = energies[j] - energies[i]
            if transition_energy > 0:
                transition_energies.append(transition_energy)
                transition_amplitudes.append(np.abs(transition_operator.matrix_element(psi_j.conj(), psi_i)) *
                                             initial_populations[i])
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
             tensor(identity(twonplus1(p.nuclear_spin)),
                    (nuclear_moment[0] * iis[0] + nuclear_moment[1] * iis[1] + nuclear_moment[2] * iis[2]))
    return hh_int


def plot_transition_amplitudes(transition_operator: Qobj, energies: Sequence[float], eigenstates: Sequence[Qobj],
                               initial_populations: np.ndarray = None, fig=None, xscale=1., xlabel=None, yscale=1.,
                               ylabel='rabi frequency (Hz)', title=None):
    """
    Plots transition amplitudes weighted by initial populations
    Args:
        transition_operator: Interaction hamiltonian for the transition
        energies: Eigenenergies of the system
        eigenstates: Eigenstates of the system
        initial_populations: Initial populations corresponding to eigenenergies, will normalize in place
        fig: matplotlib figure object to plot on (optional)
        xscale: proportional scaling for x-axis
        xlabel: label for x-axis
        yscale: proportional scaling for y-axis
        ylabel: label for y-axis
        title: plot title

    Returns:

    """
    hovertip_text = []

    for i in range(energies.shape[0]):
        for j in range(energies.shape[0]):
            if (energies[j] - energies[i]) > 0:
                hovertip_text.append(f'|{i}> --> |{j}>')

    transition_energies, transition_amplitudes = get_transition_amplitudes(transition_operator, energies, eigenstates,
                                                                           initial_populations=initial_populations)

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


def plot_nv_ground_magnetic_transition_amplitudes(transition_bvec, static_bvec, initial_populations: np.ndarray=None,
                                                  p: NVGroundParameters14N=NVGroundParameters14N()):
    """
    NV axis is in the (0, 0, 1) direction
    This has been checked against https://doi.org/10.1038/s41598-020-61669-w for the case where NV-axis is aligned to
    field.
    :param transition_bvec: RF field vector in Tesla
    :param static_bvec: Bias field vector in Tesla
    :param p: Parameters
    :return: plotly figure
    """
    p=NVGroundParameters14N()
    hh_int = get_magnetic_transition_operator(p, transition_bvec)
    energies, eigenstates = get_nv_ground_eigenspectrum(p, static_bvec)
    fig = plot_transition_amplitudes(hh_int, energies, eigenstates, xscale=1.E-6, xlabel='transition frequency (MHz)',
                                     yscale=1.E-3, ylabel='population-weighted rabi frequency (kHz)',
                                     initial_populations=initial_populations)
    fig.update_layout(title=dict(text=f'B = ({static_bvec[0] * 1.E4:.2f} G, {static_bvec[1] * 1.E4:.2f} G, '
                                      f'{static_bvec[2] * 1.E4:.2f} G)<br>'
                                      f'B_drive = ({transition_bvec[0] * 1.E4:.2f} G, '
                                      f'{transition_bvec[1] * 1.E4:.2f} G, '
                                      f'{transition_bvec[2] * 1.E4:.2f} G)'))
    fig.show()
    return fig

def lorentz_lineshape(frequencies, transition_frequency, transition_amplitude):
    """
    Gives a Lorentzian line shape for a spectrum where spontaneous relaxation processes are negligible.
    :param frequencies: frequencies, domain over which to calculate the lineshape
    :param transition_frequency: center of lorentzian peak
    :param transition_amplitude: used to get FWHM
    :return: population probability according to Lorentzian line shape
    """
    if transition_amplitude == 0:
        return np.zeros_like(frequencies)
    else:
        x = (frequencies - transition_frequency) / (transition_amplitude)
        line = 1 / (1 + np.square(x))
        return line * transition_amplitude


def get_power_broadened_spectrum(frequencies, transition_bvec, static_bvec, initial_populations,
                                  p: NVGroundParameters14N=NVGroundParameters14N()):
    """
    Gets the power-broadened spectrum for a single NV-configuration. Applied magnetic field and RF field amplitude are
    fixed.
    :param frequencies: Array of frequencies over which to calculate spectrum
    :param transition_bvec: The magnetic field vector for the magnetic transition (RF field)
    :param static_bvec: The static bias field that determines the eigenstates
    :initial_populations: Array of initial populations in eigen-energy order from ground state to highest energy
    :param p: Parameters describing the NV center
    :return: returns the probability of a transition occuring for a particular driving RF field.
    """
    hh_int = get_magnetic_transition_operator(p, transition_bvec)
    energies, eigenstates = get_nv_ground_eigenspectrum(p, static_bvec)

    transition_energies, transition_amplitudes = get_transition_amplitudes(hh_int, energies, eigenstates,
                                                                           initial_populations=initial_populations)

    spectrum = np.zeros_like(frequencies)
    for i, amp in enumerate(transition_amplitudes):
        spectrum += lorentz_lineshape(frequencies, transition_energies[i], amp)

    return spectrum

def get_power_broadened_spectrum_nv_axis_average(frequencies, transition_bvec, static_bvec, initial_state,
                                  p: NVGroundParameters14N=NVGroundParameters14N(),
                                                 nv_axes=((1,1,1), (-1,-1,1), (1,-1,-1), (-1,1,-1))):
    """
    Gets the spectral average over all possible NV axis directions (legs of tetrahedron)
    Args:
        frequencies: line space for frequencies
        transition_bvec: B-field at max amplitude for oscillating RF wave
        static_bvec: static applied B-field
        initial_state: distribution of population among ground eigenstates
        p: NV ground parameters

    Returns: spectrum to be plotted

    """
    bvecs = get_bfields(static_bvec, nv_axes[0], nv_axes[1], nv_axes[2], nv_axes[3])
    transition_bvecs = get_bfields(transition_bvec, nv_axes[0], nv_axes[1], nv_axes[2], nv_axes[3])

    spectrum = np.zeros_like(frequencies)
    for i, bvec in enumerate(bvecs):
        spectrum += get_power_broadened_spectrum(frequencies, np.array(transition_bvecs[i]), bvec,
                                                 initial_populations=initial_state)
    return spectrum / len(bvecs)


if __name__ == "__main__":
    static_bvec = 0.005 * np.array([0.4, 0.1, -0.2])
    transition_bvec = np.array([0., 0., 0.000034])
    initial_state = [1 / 3., 1 / 3., 1 / 3., 0, 0, 0, 0, 0, 0]
    frequencies = np.linspace(2.8E9, 2.95E9, num=2 ** 16)

    spectrum = get_power_broadened_spectrum_nv_axis_average(frequencies, transition_bvec, static_bvec, initial_state)
    spectrum = spectrum / max(spectrum)
    fig = plt.plot(frequencies, spectrum)
    plt.xlabel("Applied Wave Frequency (Hz)")
    plt.ylabel("Signal Strength Normalized")
    plt.title("Simulation of CW-ODMR for a Given Static and Transient Magnetic Field")
    plt.show()
