#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NEST GPU ring topology network.

Authors: Jose V.

"""

import logging
import sys

LOG = logging.getLogger(__name__)

# Used for downsampling neuron counts for plotting
import random

# Used to handle command line parameters
from argparse import ArgumentParser

# Used for storing spike data
from json import dumps
from pathlib import Path

# For timing simulation sections
from time import perf_counter_ns

# Used for error handling
from traceback import format_exc

import matplotlib.pyplot as plt
import nestgpu as ngpu
from mpi4py import MPI

parser = ArgumentParser()
parser.add_argument("--scale", type=int, default=1000)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--outpath", type=str, default=".")
args = parser.parse_args()

outpath = Path(args.outpath)
outpath.mkdir(parents=True, exist_ok=True)

SIMULATION_PARAMS = {
    "scale": args.scale,  # Order of magnitude of neuron population size
    "seed": args.seed,  # Seed for random number generation
    "pre_simtime": 100.0,  # Simulation time until reaching equilibrium in ms
    "simtime": 1000.0,  # Simulation time at equilibrium in ms
    "max_rec_spikes": 1000,  # Maximum number of spikes to record per neuron
    "verbosity": 20,  # Python logger verbosity
    "data_file": "simulation_data.json",  # Name of output file with simulation data
    "plot_spike_data": True,  # Control parameter to enable plotting
}


NETWORK_PARAMS = {
    "NE": 4 * SIMULATION_PARAMS["scale"],  # Number of excitatory neurons
    "NI": 1 * SIMULATION_PARAMS["scale"],  # Number of inhibitory neurons
    "CE": 800,  # Number of excitatory synapses per neuron
    "CI": 200,  # Number of inhibitory synapses per neuron
    "CE_REMOTE_R": 0.6,  # Ratio of total excitatory synapses from remote source
    "CI_REMOTE_R": 0.1,  # Ratio of total inhibitory synapses from remote source
}


NEURON_PARAMS = {
    "E_rev_ex": 0.0,  # mV
    "E_rev_in": -85.0,  # mV
    "tau_decay_ex": 1.0,  # ms
    "tau_decay_in": 1.0,  # ms
    "tau_rise_ex": 1.0,  # ms
    "tau_rise_in": 1.0,  # ms
}


SYNAPSE_PARAMS = {
    "weight_ex": 0.05,  # Excitatory synaptic weight in pA
    "weight_in": 0.25,  # Inhibitory synaptic weight in pA
    "remote_weight_factor": 1.4,  # Weight factor for remote connections
    "mean_delay": 0.5,  # Mean delay of delay distribution in ms
    "std_delay": 0.25,  # Standard deviation of delay distribution in ms
    "min_delay": 0.1,  # Minimum delay of delay distribution in ms
    "max_delay": 1.25,  # Maximum delay of delay distribution in ms
    "remote_delay_factor": 1.5,  # Delay distribution factor for remote connections
}


STIMULUS_PARAMS = {
    "rate": 20000.0,  # Poisson signal rate in Hz
    "weight": 0.35,  # Poisson signal amplitude in pA
    "delay": 0.2,  # Poisson signal delay in ms
    "window_stim": True,  # Whether to enable additional stimulation window
    "window_start": 400.0,  # Additional stimulus window start in ms
    "window_end": 500.0,  # Additional stimulus window end in ms
    "window_stim_fact": 0.75,  # Stimulation window rate factor
}


PLOT_PARAMS = {
    "ax_width": 18,
    "ax_height": 1,
    "fig_y_margin": 1,
    "fig_x_margin": 1,
    "title_font_size": 30,
    "label_font_size": 25,
    "legend_font_size": 25,
    "tick_font_size": 20,
    "marker_size": 1,
    "ax_y_label": "",  # MPI rank is prepended to this string
    "ax_x_label": "Spike times [ms]",
    "ax_title": "Raster plots by MPI ranks",
    "ax_x_ticks": [
        SIMULATION_PARAMS["pre_simtime"],
        STIMULUS_PARAMS["window_start"] + SIMULATION_PARAMS["pre_simtime"],
        STIMULUS_PARAMS["window_end"] + SIMULATION_PARAMS["pre_simtime"],
        SIMULATION_PARAMS["simtime"] + SIMULATION_PARAMS["pre_simtime"],
    ],
    # Number of neurons to plot can be reduced for easier plotting
    "neuron_ratio": 0.1,
    "exc_color": "blue",  # Color of spike times for excitatory neurons
    "inh_color": "red",  # Color of spike times for inhibitory neurons
    "exc_label": "Excitatory",  # Label for excitatory neuron spike times scatter
    "inh_label": "Inhibitory",  # Label for inhibitory neuron spike times scatter
    # Axes can be drawn in a single column or in individual plots
    "plot_individual": False,
    # If axes are drawn individually then each plot has a title
    # If axes are drawn together, only top level axe will have a title
    # If axes are drawn together only bottom level axe will have x label
    # If axes are drawn individually one file per axe will be created
    # in that case the rank of the process will be used as suffix for the file name
    "file_name": "ring_topology_activity",  # Plot file name
    "file_format": "png",  # Plot file format
}


def update_logging_level() -> None:
    """
    Function to initialize Python logging.
    """
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.basicConfig(level=SIMULATION_PARAMS["verbosity"], handlers=(stdout,))


def rank_log(mpi_id: int, *args):
    """
    Convenience function to add MPI rank to each logging message
    """
    args[0]("RANK %i: " + args[1], *(mpi_id, *args[2:]))


def check_stim_window(mpi_id: int) -> bool:
    """
    Function to check parameters of additional stimulus window.
    Raises and exception if parameters are inconsistent.

    Arguments:
        - mpi_id: Integer rank of MPI process.

    Returns: boolean state to enable additional window stimulus.
    """

    if not STIMULUS_PARAMS["window_stim"]:
        # Nothing is done if there is no additional window stimulus
        return False

    if STIMULUS_PARAMS["window_start"] >= SIMULATION_PARAMS["simtime"]:
        rank_log(
            mpi_id,
            LOG.critical,
            "Additional stimulus window cannot start after simulation ends",
        )
        raise Exception(mpi_id)

    if STIMULUS_PARAMS["window_end"] > SIMULATION_PARAMS["simtime"]:
        rank_log(
            mpi_id,
            LOG.critical,
            "Additional stimulus window cannot end after simulation ends",
        )
        raise Exception(mpi_id)

    if STIMULUS_PARAMS["window_start"] >= STIMULUS_PARAMS["window_end"]:
        rank_log(
            mpi_id,
            LOG.critical,
            "Additional stimulus window duration is null or negative",
        )
        raise Exception(mpi_id)

    return True


def update_stimulus(mpi_id: int, in_window: bool, local_stimulus: ngpu.NodeSeq):
    """
    Function to update stimulus during and outside of additional stimulus window.

    Arguments:
        - mpi_id: Integer rank of MPI process.
        - in_window: Boolean to indicate whether the update is done during the
                        additional stimulus window or outside.
        - local_stimulus: Node sequence of local stimulus.
    """

    # Only rank 0 performs additional stimulus window.
    # This is to ensure additional stimulus propagates across ring
    # otherwise the whole network will be stimulated and only firing
    # rate changes will be perceived.
    # Change is only applied to local stimulus.
    if mpi_id == 0:
        if in_window:
            rank_log(
                mpi_id,
                LOG.info,
                "Updating stimulus rate during additional stimulus window",
            )
            ngpu.SetStatus(
                local_stimulus,
                "rate",
                STIMULUS_PARAMS["window_stim_fact"] * STIMULUS_PARAMS["rate"],
            )
        else:
            rank_log(
                mpi_id,
                LOG.info,
                "Updating stimulus rate outside additional stimulus window",
            )
            ngpu.SetStatus(local_stimulus, "rate", STIMULUS_PARAMS["rate"])


def create_nodes(mpi_id: int, mpi_np: int) -> tuple:
    """
    Function to create neuron populations with MPI support.
    If only one process is present during simulation
    then Create is used otherwise RemoteCreate is used.

    Arguments:
        - mpi_id: Integer rank of MPI process.
        - mpi_np: Integer of total number of MPI ranks.

    Returns: a tuple composed of:
        - the list containing one node sequence per MPI rank.
        - the node sequence of the local stimulus.
    """

    if mpi_np < 1:
        rank_log(
            mpi_id,
            LOG.critical,
            "Incorrect usage of create_nodes function with %i ranks",
            mpi_np,
        )
        raise Exception(mpi_id)

    mpi_node_sequences = []
    if mpi_np > 1:
        # Create neuron populations for each MPI process.
        # This has to be done over a loop of all processes,
        # as each process needs knowledge of remotely available
        # neurons to be able to connect to it.
        # If the rank of the MPI process passed as argument to RemoteCreate
        # is different than the local process, then no neurons are
        # created, only information about remote neuron IDs is stored.
        rank_log(mpi_id, LOG.info, "Creating remote nodes")
        for i in range(mpi_np):
            mpi_node_sequences.append(
                ngpu.RemoteCreate(
                    i,  # MPI rank
                    "aeif_cond_beta",  # Neuron model
                    NETWORK_PARAMS["NE"]
                    + NETWORK_PARAMS["NI"],  # Total number of neurons
                    2,  # Number of receptors
                    NEURON_PARAMS,
                ).node_seq
            )
    else:
        rank_log(mpi_id, LOG.info, "Creating local nodes")
        mpi_node_sequences.append(
            ngpu.Create(
                "aeif_cond_beta",
                NETWORK_PARAMS["NE"] + NETWORK_PARAMS["NI"],
                2,
                NEURON_PARAMS,
            )
        )

    # Recordings can only be applied to neurons available locally
    rank_log(mpi_id, LOG.info, "Activating recording of spike times")
    ngpu.ActivateRecSpikeTimes(
        mpi_node_sequences[mpi_id], SIMULATION_PARAMS["max_rec_spikes"]
    )

    # In practice there is no mathematical difference
    # from creating a single Poisson generator or one per MPI rank
    # (an independent random number stream is created for each connection),
    # however to avoid the MPI communication overhead needed for remote connections,
    # each generator is created and connected locally.
    rank_log(mpi_id, LOG.info, "Creating Poisson generator")
    local_stimulus = ngpu.Create(
        "poisson_generator",
        1,  # 1 generator per MPI rank
        1,  # 1 receptor port per generator
        {"rate": STIMULUS_PARAMS["rate"]},
    )

    return mpi_node_sequences, local_stimulus


def connect_nodes(mpi_id: int, mpi_np: int, nodes_tuple: tuple) -> None:
    """
    Function to connect neuron populations with MPI support.
    If only one process is present during simulation
    then Connect is used otherwise RemoteConnect is used.
    When multiple MPI processes are active, network is connected
    in a ring topology i.e. each rank defines a "left" and "right"
    neighbor, each rank receives synapses from "left" neighbor,
    and sends synapses to "right" neighbor.

    Arguments:
        - mpi_id: Integer rank of MPI process.
        - mpi_np: Integer of total number of MPI ranks.
        - nodes_tuple: tuple of list of node sequence of MPI processes
                        and local stimulus.
    """

    # Using remote connection ratios we get total number of synapses
    # incoming remotely, and the rest of synapses is connected locally
    CE_REMOTE = int(NETWORK_PARAMS["CE"] * NETWORK_PARAMS["CE_REMOTE_R"])
    CI_REMOTE = int(NETWORK_PARAMS["CI"] * NETWORK_PARAMS["CI_REMOTE_R"])

    if (
        CE_REMOTE >= NETWORK_PARAMS["CE"]
        or CI_REMOTE >= NETWORK_PARAMS["CI"]
        or CE_REMOTE == 0
        or CI_REMOTE == 0
    ):
        LOG.critical(
            "Remote connection factors too high,"
            + "attempting to create %i excitatory connection and %i inhibitory connections",
            CE_REMOTE,
            CI_REMOTE,
        )
        raise Exception(mpi_id)

    # Number of local connections is safeguarded against simulating with a single process
    CE_LOCAL = NETWORK_PARAMS["CE"] - (CE_REMOTE if mpi_np > 1 else 0)
    CI_LOCAL = NETWORK_PARAMS["CI"] - (CI_REMOTE if mpi_np > 1 else 0)

    # Excitatory connections
    # connect excitatory neurons to port 0 of all neurons
    # normally distributed delays, weight and connections per neuron
    exc_local_conn_dict = {"rule": "fixed_indegree", "indegree": CE_LOCAL}
    exc_remote_conn_dict = {"rule": "fixed_indegree", "indegree": CE_REMOTE}
    exc_local_syn_dict = {
        "weight": SYNAPSE_PARAMS["weight_ex"],
        "delay": {
            "distribution": "normal_clipped",
            "mu": SYNAPSE_PARAMS["mean_delay"],
            "low": SYNAPSE_PARAMS["min_delay"],
            "high": SYNAPSE_PARAMS["max_delay"],
            "sigma": SYNAPSE_PARAMS["std_delay"],
        },
        "receptor": 0,
    }
    exc_remote_syn_dict = {
        "weight": SYNAPSE_PARAMS["remote_weight_factor"] * SYNAPSE_PARAMS["weight_ex"],
        "delay": {
            "distribution": "normal_clipped",
            "mu": SYNAPSE_PARAMS["remote_delay_factor"] * SYNAPSE_PARAMS["mean_delay"],
            "low": SYNAPSE_PARAMS["remote_delay_factor"] * SYNAPSE_PARAMS["min_delay"],
            "high": SYNAPSE_PARAMS["remote_delay_factor"] * SYNAPSE_PARAMS["max_delay"],
            "sigma": SYNAPSE_PARAMS["remote_delay_factor"]
            * SYNAPSE_PARAMS["std_delay"],
        },
        "receptor": 0,
    }

    # Inhibitory connections
    # connect inhibitory neurons to port 1 of all neurons
    # normally distributed delays, weight and connections per neuron
    inh_local_conn_dict = {"rule": "fixed_indegree", "indegree": CI_LOCAL}
    inh_remote_conn_dict = {"rule": "fixed_indegree", "indegree": CI_REMOTE}
    inh_local_syn_dict = {
        "weight": SYNAPSE_PARAMS["weight_in"],
        "delay": {
            "distribution": "normal_clipped",
            "mu": SYNAPSE_PARAMS["mean_delay"],
            "low": SYNAPSE_PARAMS["min_delay"],
            "high": SYNAPSE_PARAMS["max_delay"],
            "sigma": SYNAPSE_PARAMS["std_delay"],
        },
        "receptor": 1,
    }
    inh_remote_syn_dict = {
        "weight": SYNAPSE_PARAMS["remote_weight_factor"] * SYNAPSE_PARAMS["weight_in"],
        "delay": {
            "distribution": "normal_clipped",
            "mu": SYNAPSE_PARAMS["remote_delay_factor"] * SYNAPSE_PARAMS["mean_delay"],
            "low": SYNAPSE_PARAMS["remote_delay_factor"] * SYNAPSE_PARAMS["min_delay"],
            "high": SYNAPSE_PARAMS["remote_delay_factor"] * SYNAPSE_PARAMS["max_delay"],
            "sigma": SYNAPSE_PARAMS["remote_delay_factor"]
            * SYNAPSE_PARAMS["std_delay"],
        },
        "receptor": 1,
    }

    mpi_node_sequences, local_stimulus = nodes_tuple
    total_neurons = NETWORK_PARAMS["NE"] + NETWORK_PARAMS["NI"]
    local_neurons = mpi_node_sequences[mpi_id]
    local_NE = local_neurons[0 : NETWORK_PARAMS["NE"]]
    local_NI = local_neurons[NETWORK_PARAMS["NE"] : total_neurons]

    if mpi_np > 1:
        # Like during node creation,
        # here each MPI rank must connect by sending or receiving
        # synapses to other ranks,
        # ATTENTION: here the connection procedure requires coordination
        # from both the sender and receiver!
        # If either is not coordinated then the other will be unable to continue.
        # Examples:
        # 1) Unilateral send/receive
        #   rank 0: creates and connects local structures, then proceeds to simulate
        #   rank 1: creates and connects local structures, sends connections to rank 0
        #   -> DEADLOCK: rank 0 has no instruction to receive connections
        #                rank 0 will simulate without any interaction to rank 1
        #                rank 1 will be unable to continue
        # 2) Uncoordinated send/receive:
        #   rank 0: sends connections to rank 1
        #   rank 1: sends connections to rank 0
        #   -> DEADLOCK: no rank is trying to receive connections
        #                and no rank will be able to continue
        #                same will happen if both try to receive at the same time

        rank_log(mpi_id, LOG.info, "Connecting remote nodes")
        for source in range(mpi_np):
            # Here by checking whether the local rank is either source or target
            # at each iteration all ranks are coordinated with each other
            target = (source + 1) % mpi_np
            target_neurons = mpi_node_sequences[target]
            target_NE = target_neurons[0 : NETWORK_PARAMS["NE"]]
            target_NI = target_neurons[NETWORK_PARAMS["NE"] : total_neurons]

            # If local MPI rank is connection source
            if mpi_id == source:
                rank_log(mpi_id, LOG.info, "Sending connections to rank %i", target)
                # Connections are sent from "left" MPI rank
                ngpu.RemoteConnect(
                    source,  # Source rank (sender)
                    local_NE,  # Source population
                    target,  # Target rank (receiver)
                    target_neurons,  # Target population
                    exc_remote_conn_dict,
                    exc_remote_syn_dict,
                )
                ngpu.RemoteConnect(
                    source,  # Source rank (sender)
                    local_NI,  # Source population
                    target,  # Target rank (receiver)
                    target_neurons,  # Target population
                    inh_remote_conn_dict,
                    inh_remote_syn_dict,
                )

            # If local MPI rank is connection target
            elif mpi_id == target:
                rank_log(mpi_id, LOG.info, "Receiving connections from rank %i", source)
                # Connections are received by "right" MPI rank
                ngpu.RemoteConnect(
                    source,  # Source rank
                    target_NE,  # Source population
                    target,  # Target rank
                    local_neurons,  # Target population
                    exc_remote_conn_dict,
                    exc_remote_syn_dict,
                )
                ngpu.RemoteConnect(
                    source,  # Source rank
                    target_NI,  # Source population
                    target,  # Target rank
                    local_neurons,  # Target population
                    inh_remote_conn_dict,
                    inh_remote_syn_dict,
                )

    # Perform local connections
    rank_log(mpi_id, LOG.info, "Connecting local nodes")
    ngpu.Connect(local_NE, local_neurons, exc_local_conn_dict, exc_local_syn_dict)
    ngpu.Connect(local_NI, local_neurons, inh_local_conn_dict, inh_local_syn_dict)

    # Connect poisson generator to port 0 of all neurons
    pg_conn_dict = {"rule": "all_to_all"}
    pg_syn_dict = {
        "weight": STIMULUS_PARAMS["weight"],
        "delay": STIMULUS_PARAMS["delay"],
        "receptor": 0,
    }

    rank_log(mpi_id, LOG.info, "Connecting local Poisson generator")
    ngpu.Connect(local_stimulus, local_neurons, pg_conn_dict, pg_syn_dict)


def gather_spike_times(mpi_id: int, local_node_sequence: ngpu.NodeSeq) -> dict:
    """
    Function to collect spike times from each MPI rank.
    Each rank locally collects recorded spike times and then
    communicates them to rank 0 for aggregation.

    Arguments:
        - mpi_id: Integer rank of MPI process.
        - mpi_np: Integer of total number of MPI ranks.
        - local_node_sequence: Node sequence of local MPI rank.

    Returns:
        - dictionary containing MPI rank specific spike data.
    """

    # get local spikes
    rank_log(mpi_id, LOG.info, "Collecting local spike times")
    spike_times = ngpu.GetRecSpikeTimes(local_node_sequence)

    # Sort neurons by population
    rank_log(mpi_id, LOG.info, "Sorting local spike times")
    exc_spike_count = 0
    exc_spike_times = []
    inh_spike_count = 0
    inh_spike_times = []
    for idx_neuron in range(NETWORK_PARAMS["NE"] + NETWORK_PARAMS["NI"]):
        spikes = spike_times[idx_neuron]
        if len(spikes) > 0:
            # Identify neuron population by ID range
            if NETWORK_PARAMS["NE"] > idx_neuron:
                neuron_spike_times = (idx_neuron, [])
                for time in spikes:
                    # Only get spike times after pre-simulation
                    if time > SIMULATION_PARAMS["pre_simtime"]:
                        exc_spike_count += 1
                        neuron_spike_times[1].append(time)
                if len(neuron_spike_times[1]) > 0:
                    exc_spike_times.append(neuron_spike_times)
            else:
                neuron_spike_times = (idx_neuron, [])
                for time in spikes:
                    if time > SIMULATION_PARAMS["pre_simtime"]:
                        inh_spike_count += 1
                        neuron_spike_times[1].append(time)
                if len(neuron_spike_times[1]) > 0:
                    inh_spike_times.append(neuron_spike_times)

    # Compute time averaged firing rates
    time_averaged_firing_rates = {
        "E": exc_spike_count
        / (NETWORK_PARAMS["NE"] * SIMULATION_PARAMS["simtime"])
        * 1e3,
        "I": inh_spike_count
        / (NETWORK_PARAMS["NI"] * SIMULATION_PARAMS["simtime"])
        * 1e3,
    }
    rank_log(
        mpi_id,
        LOG.info,
        "Local time averaged firing rate by population: %s",
        dumps(time_averaged_firing_rates),
    )

    # Combine data in a single dictionary
    spike_data = {
        "rank": mpi_id,
        "time_averaged_firing_rates": time_averaged_firing_rates,
        "spike_times": {"E": exc_spike_times, "I": inh_spike_times},
    }

    return spike_data


def plot_spike_data(mpi_id: int, mpi_spike_data: list) -> None:
    """
    Function to plot spike data aggregated by MPI rank.
    Data structure here expects a list of dictionaries.
    One dictionary per MPI rank, each dictionary containing
    a dictionary for spike times aggregated by neuron and
    by population.
    A raster plot showing a percentage of total neurons
    by population is drawn for each MPI rank in a vertical
    column.
    A small diagram of the structure of the data is
    shown here:
        list(
            dictionary(
                "rank": MPI rank as an integer
                "spike_times": dictionary(
                    "E" : list(
                        tuple(
                            neuron index as integer,
                            list(
                                spike times as floats
                            )
                        ),
                        ...
                    ),
                    "I": list(
                        ...
                    )
                )
            ),
            ...
        )

    Arguments:
        - mpi_id: Integer rank of MPI process.
        - mpi_spike_data: List of dictionaries containing
                            MPI rank specific spike data
    """

    rank_log(mpi_id, LOG.info, "Plotting MPI spike data")

    # We sort the rank data by rank
    sorted_rank_data = sorted(mpi_spike_data, key=lambda x: x["rank"])
    num_ranks = len(sorted_rank_data)

    if not PLOT_PARAMS["plot_individual"]:
        fig, axs = plt.subplots(
            nrows=num_ranks,
            ncols=1,
            figsize=(
                2 * PLOT_PARAMS["fig_x_margin"] + PLOT_PARAMS["ax_width"],
                (num_ranks + 1) * PLOT_PARAMS["fig_y_margin"]
                + num_ranks * PLOT_PARAMS["ax_height"],
            ),
        )

    for rd_index, rank_data in enumerate(sorted_rank_data):

        if PLOT_PARAMS["plot_individual"]:
            fig, ax = plt.subplots(
                figsize=(
                    2 * PLOT_PARAMS["fig_x_margin"] + PLOT_PARAMS["ax_width"],
                    2 * PLOT_PARAMS["fig_y_margin"] * PLOT_PARAMS["ax_height"],
                )
            )
        elif num_ranks > 1:
            ax = axs[rd_index]
        else:
            ax = axs

        exc_spike_times = rank_data["spike_times"]["E"]
        inh_spike_times = rank_data["spike_times"]["I"]
        exc_sample_size = len(exc_spike_times)
        inh_sample_size = len(inh_spike_times)
        num_exc_to_plot = int(exc_sample_size * PLOT_PARAMS["neuron_ratio"])
        num_inh_to_plot = int(inh_sample_size * PLOT_PARAMS["neuron_ratio"])

        if (
            num_exc_to_plot > exc_sample_size
            or num_inh_to_plot > inh_sample_size
            or num_exc_to_plot == 0
            or num_inh_to_plot == 0
        ):
            rank_log(
                mpi_id,
                LOG.error,
                "Cannot plot neuron ratio %f",
                PLOT_PARAMS["neuron_ratio"],
            )
            num_exc_to_plot = exc_sample_size
            num_inh_to_plot = inh_sample_size

        if num_exc_to_plot < exc_sample_size:
            exc_spike_times = random.sample(exc_spike_times, num_exc_to_plot)
        if num_inh_to_plot < inh_sample_size:
            inh_spike_times = random.sample(inh_spike_times, num_inh_to_plot)

        for exc_index, (neuron_id, spike_times) in enumerate(exc_spike_times):
            ax.scatter(
                spike_times,
                [neuron_id] * len(spike_times),
                s=PLOT_PARAMS["marker_size"],
                color=PLOT_PARAMS["exc_color"],
                label=(PLOT_PARAMS["exc_label"] if exc_index == 0 else ""),
                rasterized=True,
            )

        for inh_index, (neuron_id, spike_times) in enumerate(inh_spike_times):
            ax.scatter(
                spike_times,
                [neuron_id] * len(spike_times),
                s=PLOT_PARAMS["marker_size"],
                color=PLOT_PARAMS["inh_color"],
                label=(PLOT_PARAMS["inh_label"] if inh_index == 0 else ""),
                rasterized=True,
            )

        if PLOT_PARAMS["plot_individual"] or rd_index == 0:
            ax.set_title(
                PLOT_PARAMS["ax_title"], fontsize=PLOT_PARAMS["title_font_size"]
            )

        ax.set_ylabel(
            f"MPI rank {rank_data['rank']}" + PLOT_PARAMS["ax_y_label"],
            fontsize=PLOT_PARAMS["label_font_size"],
        )
        ax.set_yticks([])

        ax.tick_params(labelsize=PLOT_PARAMS["tick_font_size"])
        if not PLOT_PARAMS["plot_individual"] and rd_index < num_ranks - 1:
            ax.set_xlabel("")
            ax.set_xticks([])
        else:
            ax.set_xlabel(
                PLOT_PARAMS["ax_x_label"], fontsize=PLOT_PARAMS["label_font_size"]
            )
            ax.set_xticks(PLOT_PARAMS["ax_x_ticks"])

        if PLOT_PARAMS["plot_individual"]:
            plot_file = (
                outpath
                / f"{PLOT_PARAMS['file_name']}_rank_{rank_data['rank']}.{PLOT_PARAMS['file_format']}"
            )
            fig.tight_layout()
            fig.savefig(plot_file)

    if not PLOT_PARAMS["plot_individual"]:
        plot_file = outpath / f"{PLOT_PARAMS['file_name']}.{PLOT_PARAMS['file_format']}"
        fig.tight_layout()
        fig.savefig(plot_file)


def main() -> None:
    time = perf_counter_ns()
    # Get local rank ID and total number of ranks
    mpi_id = ngpu.HostId()
    mpi_np = ngpu.HostNum()
    do_window_stimulus = check_stim_window(mpi_id)

    # Update Python logging verbosity
    # and set kernel parameter for random number generation seed
    update_logging_level()
    ngpu.SetKernelStatus(
        {
            "verbosity_level": 0,
            "rnd_seed": SIMULATION_PARAMS["seed"],
        }
    )
    time_init = (perf_counter_ns() - time) / 1e9
    rank_log(mpi_id, LOG.info, "Initialization time %f", time_init)
    del time

    # Construct network
    # Tuple is composed of
    #   - mpi_node_sequence (item 0)
    #   - local_stimulus_node (item 1)
    time = perf_counter_ns()
    nodes_tuple = create_nodes(mpi_id, mpi_np)
    time_create = (perf_counter_ns() - time) / 1e9
    rank_log(mpi_id, LOG.info, "Creation time %f", time_create)
    del time

    time = perf_counter_ns()
    connect_nodes(mpi_id, mpi_np, nodes_tuple)
    time_connect = (perf_counter_ns() - time) / 1e9
    rank_log(mpi_id, LOG.info, "Connection time %f", time_connect)
    del time

    # Explicit call to calibrate to finalize network construction
    time = perf_counter_ns()
    rank_log(mpi_id, LOG.info, "Calibrating")
    ngpu.Calibrate()
    time_calibrate = (perf_counter_ns() - time) / 1e9
    rank_log(mpi_id, LOG.info, "Calibration time %f", time_calibrate)
    del time

    # Simulate
    time = perf_counter_ns()
    rank_log(mpi_id, LOG.info, "Running pre-simulation")
    ngpu.Simulate(SIMULATION_PARAMS["pre_simtime"])
    time_presim = (perf_counter_ns() - time) / 1e9
    rank_log(mpi_id, LOG.info, "Pre-simulation time %f", time_presim)
    del time

    time = perf_counter_ns()
    if do_window_stimulus:
        rank_log(
            mpi_id, LOG.info, "Running simulation before additional stimulus window"
        )
        ngpu.Simulate(STIMULUS_PARAMS["window_start"])
        update_stimulus(mpi_id, True, nodes_tuple[1])
        rank_log(
            mpi_id, LOG.info, "Running simulation during additional stimulus window"
        )
        ngpu.Simulate(STIMULUS_PARAMS["window_end"] - STIMULUS_PARAMS["window_start"])
        update_stimulus(mpi_id, False, nodes_tuple[1])
        rank_log(
            mpi_id, LOG.info, "Running simulation after additional stimulus window"
        )
        ngpu.Simulate(SIMULATION_PARAMS["simtime"] - STIMULUS_PARAMS["window_end"])
    else:
        rank_log(mpi_id, LOG.info, "Running simulation")
        ngpu.Simulate(SIMULATION_PARAMS["simtime"])
    time_sim = (perf_counter_ns() - time) / 1e9
    rank_log(mpi_id, LOG.info, "Simulation time %f", time_sim)
    del time

    # Every rank has its own spike data
    time = perf_counter_ns()
    mpi_spike_data = gather_spike_times(mpi_id, nodes_tuple[0][mpi_id])
    time_gather = (perf_counter_ns() - time) / 1e9
    rank_log(mpi_id, LOG.info, "Gather spikes time %f", time_gather)
    del time

    # Here we gather all data from MPI ranks at rank 0
    rank_log(mpi_id, LOG.info, "Gathering spike data at rank 0")
    mpi_spike_data = MPI.COMM_WORLD.gather(mpi_spike_data, root=0)
    # Now only MPI rank 0 posses the spike data
    # For other ranks the variable mpi_spike_data points to None

    # Here we collect all timers measured in the current MPI process
    mpi_times = {
        "rank": mpi_id,
        "time_init": time_init,
        "time_create": time_create,
        "time_connect": time_connect,
        "time_calibrate": time_calibrate,
        "time_presim": time_presim,
        "time_sim": time_sim,
        "time_gather": time_gather,
    }

    # Like for spike data, we gather everything at rank 0
    rank_log(mpi_id, LOG.info, "Gathering timer data at rank 0")
    mpi_times = MPI.COMM_WORLD.gather(mpi_times, root=0)

    # This section is only done by MPI rank 0
    # other MPI processes will now exit and free resources
    if mpi_id == 0:
        # Here we append the parameters used during simulation
        # for reproducibility and reusability purposes
        output_data = {
            "params": {
                "simulation_params": SIMULATION_PARAMS,
                "network_params": NETWORK_PARAMS,
                "neuron_params": NEURON_PARAMS,
                "synapse_params": SYNAPSE_PARAMS,
                "stimulus_params": STIMULUS_PARAMS,
                "plot_params": PLOT_PARAMS,
            },
            "mpi_times": mpi_times,
            "mpi_spike_data": mpi_spike_data,
        }
        rank_log(
            mpi_id,
            LOG.info,
            "Storing simulation data to %s",
            SIMULATION_PARAMS["data_file"],
        )
        (outpath / SIMULATION_PARAMS["data_file"]).write_text(dumps(output_data))
        if SIMULATION_PARAMS["plot_spike_data"]:
            plot_spike_data(mpi_id, mpi_spike_data)


if __name__ == "__main__":
    # ATTENTION: When simulating with MPI
    # explicit call to MpiInit MUST be performed
    # before calling any MPI functions
    ngpu.ConnectMpiInit()

    try:
        main()
    except Exception:
        # In case an exception happens at the local rank level
        # abort operations of all other ranks
        LOG.critical(format_exc())
        MPI.COMM_WORLD.Abort(1)

    # ATTENTION: When simulating with MPI
    # explicit call to MpiFinalize MUST be performed
    # otherwise processes will exit returning an error
    ngpu.MpiFinalize()
