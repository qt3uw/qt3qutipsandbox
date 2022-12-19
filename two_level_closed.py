from dataclasses import dataclass
from typing import List, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
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


@dataclass
class TwoLevelUnitaryRamseySequenceParameters(TwoLevelUnitaryMonochromePulseParameters):
    """
    This is a parameter class for simulating a Ramsey sequence on a two-level system.
    Attributes:
        f_rabi: Rabi frequency on resonance for the coupling field
        f_splitting: Frequency corresponding to the two-level splitting energy
        f_drive: Frequency of oscillation of the driving field
        initial_state: Initial state coefficients, initial_state[0]|0> + initial_state[1]|1>
        pulse_duration: Duration of the driving field pulse
        free_evolution_time: Duration of the Ramsey evolution time (between pulses)
        dt: Timestep used to generate the t_list of the qutip.sesolve method
    """
    free_evolution_time: float = 10 / 1.E9


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


def run_two_level_unitary_ramsey_sequence(p: TwoLevelUnitaryRamseySequenceParameters) -> qt.Qobj:
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

    # First pulse
    if p.pulse_duration < p.dt:
        raise ValueError('All pulse durations must be greater than the simulation time step')
    pulse_times = np.arange(0, p.pulse_duration, p.dt)
    res = qt.sesolve(hh, psi_0, pulse_times, args=args)
    state_after_first_pulse = res.states[-1]

    # Free evolution
    free_evolution_times = np.arange(0, p.free_evolution_time, p.dt)
    res = qt.sesolve(h0, state_after_first_pulse, free_evolution_times)
    state_after_free_evolution = res.states[-1]

    # Second pulse
    res = qt.sesolve(hh, state_after_free_evolution, pulse_times, args=args)
    final_state = res.states[-1]
    return final_state


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


def plot_two_level_ramsey_evolution(p: TwoLevelUnitaryRamseySequenceParameters, ramsey_durations: Iterable,
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
    nn = np.zeros_like(ramsey_durations)
    for idx, ramsey_duration in enumerate(ramsey_durations):
        print(f'running drive frequency {idx}/{ramsey_durations.shape[0]}')
        p.free_evolution_time = ramsey_duration
        res = run_two_level_unitary_ramsey_sequence(p)
        nn[idx] = qt.expect(qt.num(2), res)
    if fig is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(ramsey_durations * 1.E9, nn, **plotkwargs)
    ax.set_ylabel('<n>')
    ax.set_xlabel('free evolution time (ns)')
    ax.grid()
    return fig, ax


def plot_two_level_ramsey_spectrum(p: TwoLevelUnitaryRamseySequenceParameters, splittings: Iterable,
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
    nn = np.zeros_like(splittings)
    for idx, splitting in enumerate(splittings):
        print(f'running drive frequency {idx}/{splittings.shape[0]}')
        p.f_splitting = splitting
        res = run_two_level_unitary_ramsey_sequence(p)
        nn[idx] = qt.expect(qt.num(2), res)
    if fig is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot((p.f_drive - splittings) * 1.E-6, nn, **plotkwargs)
    ax.set_ylabel('<n>')
    ax.set_xlabel('detuning (MHz)')
    ax.grid()
    return fig, ax


def get_check_parameters() -> TwoLevelUnitaryRamseySequenceParameters:
    """
    Gets example parameters set to be used in examples.
    :return: Example parameters
    """
    p = TwoLevelUnitaryRamseySequenceParameters(f_rabi=1.E6,
                                                f_splitting=1.E9,
                                                f_drive=1.E9,
                                                initial_state=np.array([1, 0]),
                                                pulse_duration=0.5 / 1.E6,
                                                free_evolution_time=1. / 1.E6,
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
    colors = get_sequential_colormap(len(f_rabis))
    for idx, f_rabi in enumerate(f_rabis):
        p.f_rabi = f_rabi
        pulse_durations = np.linspace(p.dt, 2 / p.f_rabi, num)
        fig, ax = plot_two_level_monochrome_drive_population_dynamics(p, pulse_durations, fig=fig,
                                                                      ax=ax, label=f'f_rabi = {f_rabi*1.E-6:.2f} MHz',
                                                                      color=colors[idx])
    ax.grid()
    fig.legend()
    fig.savefig('check_resonant_two_level_drive_population_dynamics.pdf')

def check_pi_pulse_spectrum(f_rabi=1.E6, scanwidth=10.E6, num=51):
    """
    Plots spectrum for square pulse
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


def check_ramsey_spectrum(f_splitting=1.E9, scanwidth=1.E6, num=101, dt=1.E-8):
    """

    :param f_splitting: Center two-level energy splitting to use
    :param scanwidth: Full width of the splitting frequency scan.
    :param num:
    :param dt:
    :return:
    """
    free_evolution_times = [1.E-6, 3.E-6, 1.E-5, 3.E-5]
    p = get_check_parameters()
    p.dt = dt
    p.pulse_duration = 0.25 / p.f_rabi
    p.f_splitting = f_splitting
    p.f_drive = f_splitting
    colors = get_sequential_colormap(len(free_evolution_times))
    fig, ax = plt.subplots(1, 1)
    for idx, free_evolution_time in enumerate(free_evolution_times):
        p.free_evolution_time = free_evolution_time
        splittings = np.linspace(f_splitting - 0.5 * scanwidth, f_splitting + 0.5 * scanwidth, num=num)
        fig, ax = plot_two_level_ramsey_spectrum(p, splittings=splittings, fig=fig, ax=ax, linestyle='None', marker='o',
                                                 label=f'{p.free_evolution_time * 1.E6} us',
                                                 color=colors[idx])
    ax.set_title(f'f_splitting = {f_splitting * 1.E-9:.2f} GHz')
    fig.legend()
    fig.savefig(f'check_ramsey_spectrum_f_splitting_{f_splitting*1.E-9:.2f}.pdf')


def check_ramsey_evolution(min_duration=10.E-9, max_duration=12.E-9, dt=0.1E-9, f_rabi=3.E7):
    p = get_check_parameters()
    p.dt = dt
    p.f_rabi = f_rabi
    p.pulse_duration = 0.25 / p.f_rabi
    ramsey_durations = np.arange(min_duration, max_duration, p.dt)
    fig, ax = plot_two_level_ramsey_evolution(p, ramsey_durations=ramsey_durations, linestyle='None', marker='o')
    fig.savefig('check_ramsey_evolution.pdf')



if __name__ == "__main__":
    # check_ramsey_spectrum(f_splitting=1.E8, dt=1.E-8)
    # check_ramsey_spectrum(f_splitting=1.E9, dt=1.E-8)
    # check_ramsey_spectrum(f_splitting=2.88E9, dt=0.3E-9)
    check_ramsey_evolution()
    # check_resonant_two_level_drive_population_dynamics()
    # check_pi_pulse_spectrum(f_rabi=3.0E6)
    # check_pi_pulse_spectrum(f_rabi=1.0E6)
    # check_pi_pulse_spectrum(f_rabi=0.3E6)
    # check_pi_pulse_spectrum(f_rabi=0.1E6)

    plt.show()
    # check_resonant_two_level_drive_population_dynamics()
