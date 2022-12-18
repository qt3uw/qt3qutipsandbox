from dataclasses import dataclass
from typing import List, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import h, hbar
import qutip as qt

from plotting import get_sequential_colormap


@dataclass
class TwoLevelUnitaryMonochromePulseParameters:
    """
    This is a parameter class for simulating a monochrome pulse on a two-level system.
    Attributes:
        f_rabi: Rabi frequency on resonance for the coupling field
        f_splitting: Frequency corresponding to the two-level splitting energy
        f_drive: Frequency of oscillation of the driving field
        initial_state: Initial state coefficients, initial_state[0]|0> + initial_state[1]|1>
        pulse_duration: Duration of the driving field pulse
        dt: Timestep used to generate the t_list of the qutip.sesolve method
    """
    f_rabi: float = 1.E6
    f_splitting: float = 1.E9
    f_drive: np.ndarray = np.linspace(0.8E9, 1.2E9, num=100)
    initial_state: np.ndarray = np.array([1, 0])
    pulse_duration: float = 0.5 / 1.E6
    dt: float = 1. / 1.E9


def run_two_level_unitary_monochrome_pulse(p: TwoLevelUnitaryMonochromePulseParameters) -> qt.Qobj:
    """
    Closed system eigenstates
    :param p: Parameters describing the two-level system
    :return: final state vector
    """
    vac = qt.basis(2, 0)
    one = qt.basis(2, 1)
    psi_0 = p.initial_state[0] * vac + p.initial_state[1] * one
    args = {'w': p.f_drive * 2 * np.pi}
    h0 = 0.5 * qt.sigmaz() * 2 * np.pi * p.f_splitting + 0.5 * 2 * np.pi * p.f_splitting * qt.identity(2)
    h1 = - p.f_rabi * 2 * np.pi * qt.sigmax()
    hh = [h0, [h1, 'sin(w * t)']]
    if p.pulse_duration < p.dt:
        raise ValueError('All pulse durations must be greater than the simulation time step')
    times = np.arange(0, p.pulse_duration, p.dt)
    res = qt.sesolve(hh, psi_0, times, args=args)
    return res.states[-1]

def plot_two_level_monochrome_drive_population_dynamics(p: TwoLevelUnitaryMonochromePulseParameters, pulse_durations: Iterable,
                                                        fig: plt.Figure=None, ax: plt.Axes=None,
                                                        **plotkwargs):
    """
    :param p: Parameters
    :param t_lim:
    :param timestep: timestep used in
    :param fig: figure object for plotting over another figure
    :param ax: Axes object for plotting over another figure
    :param plotkwargs: kwargs dictionary for the matplotlib.axes.Axes.plot method
    :return:
    """
    nn = np.zeros_like(pulse_durations)
    for idx, t in enumerate(pulse_durations):
        print(f'running pulse duration {idx}/{pulse_durations.shape[0]}')
        p.pulse_duration = t
        res = run_two_level_unitary_monochrome_pulse(p)
        nn[idx] = qt.expect(qt.num(2), res)
    if fig is None:
        fig, ax = plt.subplots(1, 1)
    if 'marker' in plotkwargs.keys():
        plotkwargs.pop()
        print('Warning: plot marker is overridden by plot_two_level_drive_population_dynamics')
    ax.plot(pulse_durations * 1.E6, nn, marker='o', **plotkwargs)
    ax.set_ylabel('<n>')
    ax.set_xlabel('time (us)')
    return fig, ax


def plot_two_level_monochrome_drive_spectrum(p: TwoLevelUnitaryMonochromePulseParameters, drive_frequencies: Iterable,
                                             fig: plt.Figure=None, ax: plt.Axes=None,
                                             **plotkwargs):
    """
    :param p: Parameters
    :param t_lim:
    :param timestep: timestep used in
    :param fig: figure object for plotting over another figure
    :param ax: Axes object for plotting over another figure
    :param plotkwargs: kwargs dictionary for the matplotlib.axes.Axes.plot method
    :return: fig, ax
    """
    nn = np.zeros_like(drive_frequencies)
    for idx, f in enumerate(drive_frequencies):
        print(f'running drive frequency {idx}/{drive_frequencies.shape[0]}')
        p.f_drive = f
        res = run_two_level_unitary_monochrome_pulse(p)
        nn[idx] = qt.expect(qt.num(2), res)
    if fig is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot((drive_frequencies - p.f_splitting) * 1.E-6, nn, **plotkwargs)
    ax.set_ylabel('<n>')
    ax.set_xlabel('detuning (MHz)')
    ax.grid()
    return fig, ax


def get_check_parameters() -> TwoLevelUnitaryMonochromePulseParameters:
    p = TwoLevelUnitaryMonochromePulseParameters(f_rabi=1.E6,
                                                 f_splitting=1.E9,
                                                 f_drive=1.E9,
                                                 initial_state=np.array([1, 0]),
                                                 pulse_duration=0.5 / 1.E6,
                                                 dt=1. / 1.E8)
    return p

def check_resonant_two_level_drive_population_dynamics(num=51):
    """
    1 MHz rabi frequency on resonance with a 1 GHz splitting, scan pulse duration over two Rabi periods
    :param num: Number of pulse durations to compute over two rabi periods
    :return:
    """
    p = get_check_parameters()
    f_rabis = [0.25E6, 1.E6, 4.E6]
    fig, ax = plt.subplots(1, 1)
    for f_rabi in f_rabis:
        p.f_rabi = f_rabi
        pulse_durations = np.linspace(p.dt, 2 / p.f_rabi, num)
        fig, ax = plot_two_level_monochrome_drive_population_dynamics(p, pulse_durations, fig=fig,
                                                                      ax=ax, label=f'f_rabi = {f_rabi*1.E-6:.2f} MHz')
    ax.grid()
    fig.legend()
    fig.savefig('check_resonant_two_level_drive_population_dynamics.pdf')

def check_pi_pulse_spectrum(f_rabi=1.E6, scanwidth=10.E6, num=51):
    """
    1 MHz rabi frequency pi pulse, 100 MHz splitting, scanning detuning over scanwidth
    :param scanwidth: Full width of frequency scan
    :param num: Number of frequencies to scan over
    :return
    """
    p = get_check_parameters()
    p.f_rabi = f_rabi
    frequencies = np.linspace(p.f_drive - 0.5 * scanwidth, p.f_drive + 0.5 * scanwidth, num)
    fig, ax = plt.subplots(1, 1)
    n_pis = [0.2, 0.5, 1.0, 5.0, 20.0]
    colors = get_sequential_colormap(num=len(n_pis))
    for i, n_pi in enumerate(n_pis):
        p.pulse_duration = n_pi * 0.5 / p.f_rabi
        fig, ax = plot_two_level_monochrome_drive_spectrum(p, drive_frequencies=frequencies, marker='.',
                                                           fig=fig, ax=ax, label=f'{n_pi}-pi pulse', color=colors[i])
    ax.set_title(f'f_rabi = {p.f_rabi * 1.E-6:.2f} MHz')
    fig.legend()
    fig.savefig(f'check_pi_pulse_spectrum_f_rabi_{f_rabi * 1.E-6:.1f}_MHz.pdf')



if __name__ == "__main__":
    check_resonant_two_level_drive_population_dynamics()
    # check_pi_pulse_spectrum(f_rabi=3.0E6)
    # check_pi_pulse_spectrum(f_rabi=1.0E6)
    # check_pi_pulse_spectrum(f_rabi=0.3E6)
    # check_pi_pulse_spectrum(f_rabi=0.1E6)

    plt.show()
    #check_resonant_two_level_drive_population_dynamics()


