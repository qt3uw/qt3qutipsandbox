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


def rotate_about_z(vec, phi):
    vec = [vec[0]*np.cos(phi) - vec[1]*np.sin(phi), vec[0]*np.sin(phi) + vec[1]*np.cos(phi), vec[2]]
    return vec


def rodrigues(vec, k, theta):
    """
    The Rodrigues rotation calculation for a given vector bvec about unit vector k by angle theta.
    :param vec: vector to be rotated
    :param k: axis of rotation
    :param theta: rotation angle
    :return: rotated vector
    """
    crossp = [k[1]*vec[2], -k[0]*vec[2], k[0]*vec[1] - k[1]*vec[0]]
    dotp = k[0]*vec[0] + k[1]*vec[1] + k[2]*vec[2]
    fin = np.array(vec)*np.cos(theta) + np.array(crossp)*np.sin(theta) + np.array(k)*dotp*(1-np.cos(theta))

    return [fin[0], fin[1], fin[2]]


def bvec_rotation(bvec, nvvec):
    """
    Rotates a magnetic field vector such that a given NV-axis is aligned along (0,0,1).
    Acts to rotate the coordinate system.
    :param bvec: magnetic field vector in Cartesian
    :param nnvec: new NV-configuration vector in Cartesian
    :return: rotated magnetic field vector
    """
    k1 = nvvec[1]/(nvvec[0]**2 + nvvec[1]**2)**0.5
    k2 = -nvvec[0]/(nvvec[0]**2 + nvvec[1]**2)**0.5
    ang = np.arccos(nvvec[2]/(np.linalg.norm(nvvec))) # rotation angle

    return rodrigues(bvec, [k1, k2, 0], ang) # rotation about unit vector k


def get_bfields(bvec, phi):
    """
    Gives magnetic field vectors for all NV configurations. Standard NV-axis configuration is given by [111], [1-1-1],
    [-11-1], and [-1-11], but this axes set can be rotated about the z-axis by provided angle phi.
    :param bvec: magnetic field in Cartesian
    :param phi: azimuthal orientation of NV system (Cartesian)
    :return: an array of magnetic field vectors w.r.t. all four NV-axis oriented coordinate systems.
    """
    b1 = bvec_rotation(bvec, rotate_about_z([1,1,1], phi))
    b2 = bvec_rotation(bvec, rotate_about_z([1,-1,-1], phi))
    b3 = bvec_rotation(bvec, rotate_about_z([-1,1,-1], phi))
    b4 = bvec_rotation(bvec, rotate_about_z([-1,-1,1], phi))

    return [b1, b2, b3, b4]


def get_nv_ground_hamiltonian(p:NVGroundParameters14N) -> Qobj:
    """
    Gets ground state hamiltonian in frequency units (h=1) with representation ordering S, I
    From Doherty et a., Physics Reports 528 (2013)
    Nitrogen assumed as N-14.
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
                                                                                identity(twonplus1(p.nuclear_spin)) *
                                                                                (nnplus1(p.nuclear_spin) / 3)))
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
    Plots of all of the transition amplitudes
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

def lorentz_lineshape(lspace, transition_energy, transition_amplitude):
    """
    Gives a Lorentzian lineshape for a spectrum where spontaneous relaxation processes are negligible.
    :param lspace: linespace
    :param transition_energy: acts as position of maximum
    :param transition_amplitude: used to get FWHM
    :return: population probability according to Lorentzian line shape
    """
    x = lspace ##unnecessary
    if transition_amplitude == 0:
        return np.zeros_like(lspace)
    else:
        x = (x - transition_energy)/(transition_amplitude) # Relaxation constant = transition amplitude
        line = 1/(2*(1+np.square(x)))
        return line


def get_power_broadened_spectrum(transition_bvec, static_bvec, initial_state,
                                  p: NVGroundParameters14N=NVGroundParameters14N(), yscale=1, res = 1024):
    """
    Gets the power-broadened spectrum for a single NV-configuration. Applied magnetic field and RF field amplitude are
    fixed.
    :param transition_bvec: The magnetic field given by electromagnetic wave.
    :param static_bvec: The applied magnetic field
    :param p: Parameters describing the NV center
    :param res: resolution
    :return: returns the probability of a transition occuring for a particular driving RF field.
    """
    p=NVGroundParameters14N()
    hh_int = get_magnetic_transition_operator(p, transition_bvec)
    energies, eigenstates = get_nv_ground_eigenspectrum(p, static_bvec)

    transition_energies, transition_amplitudes = get_transition_amplitudes(hh_int, energies, eigenstates)
    transition_energies = transition_energies*yscale
    transition_amplitudes = transition_amplitudes*yscale

    lspace = np.linspace(2500, 4000, num = res) # Line space can be manually altered here
    spectrum = np.zeros_like(lspace)
    # Apply lineshape
    for i in range(9, 0, -1):
        hold = np.zeros_like(lspace)
        for j in range(i - 1):
            hold += 0.5*lorentz_lineshape(lspace, transition_energies[j], transition_amplitudes[j]*2)
            transition_energies = np.delete(transition_energies, 0)
            transition_amplitudes = np.delete(transition_amplitudes, 0)
        if max(hold) != 0:
            spectrum += hold*initial_state[i - 1]/max(hold)
            #print(max(spectrum))

    return lspace, spectrum


def plot_nv_config_averaging_power_broad(transition_bvec, static_bvec, initial_state,
                                  p: NVGroundParameters14N=NVGroundParameters14N(), xlabel='Transition frequency (MHz)',
                                  ylabel='Percentage of transitioning population', title=None, res = 1024):
    bvecs = get_bfields(static_bvec, 0) # Phi changed manually-- set here to 0

    lspace, spectrum = get_power_broadened_spectrum(transition_bvec, bvecs[0], initial_state, yscale=1.E-6)
    for i in range(1, 4):
        spectrum += get_power_broadened_spectrum(transition_bvec, bvecs[i], initial_state, yscale=1.E-6)[1]
    spectrum = spectrum/4

    fig = plt.plot(lspace, spectrum)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == "__main__":


    bmag = 4.6E-3
    phi = 0. * np.pi / 180. # Polar angle
    theta = 0.
    static_bvec = bmag * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
    transition_bvec = np.array([0., 1.E-4, 0.E-4])
    initial_state = [0, 0, 0.6, 0, 0, 0, 0.4, 0, 0]

    #plot_nv_ground_magnetic_transition_amplitudes(transition_bvec=transition_bvec,
    #                                              static_bvec=static_bvec)
    #plot_nv_ground_eigenspectrum(NVGroundParameters14N(), bvector=static_bvec)


    #lspace, spectrum = get_power_broadened_spectrum(transition_bvec=transition_bvec, static_bvec=static_bvec,
    #                                                initial_state=initial_state, tescale=1.E-6, tascale=1.E-3)
    #plt.plot(lspace, spectrum)
    #plt.show()
    #plot_nv_config_averaging_power_broad(transition_bvec=transition_bvec, static_bvec=static_bvec,
    #                                     initial_state=initial_state)
    freq = np.linspace(0, 10, 1024)
    testline = lorentz_lineshape(freq, 5, 2)
    plt.plot(freq, testline, label = "FWHM = 1")

    testline = lorentz_lineshape(freq, 5, 1)
    plt.plot(freq, testline, label = "FWHM = 1/2")


    testline = lorentz_lineshape(freq, 5, 3)
    plt.plot(freq, testline, label = "FWHM = 3/2")
    #plt.xlabel("transition energy MHz")
    #plt.ylabel("population transition probability")
    plt.legend()
    plt.show()


