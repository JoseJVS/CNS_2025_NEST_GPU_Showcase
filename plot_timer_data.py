#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NEST GPU ring topology benchmark data plot script.

This script was designed to plot the benchmarking data
of the ring topology network.

The benchmark was performed on 1 and 2 computing nodes
from LEONARDO Booster.
Using 1 to 8 GPUs, the ring network was simulated at
1, 10, 100, 1000, 10000, or 100000 scale.
Each combination ran for 10 different simulation seeds.
In total, 480 simulation runs were performed.

This script takes as input the aggregated timer data
from all simulation runs.
Details on the structure of the data are provided in
the docstring of load_and_flatten_data function.

Authors: Jose V.

"""
import logging
import sys

LOG = logging.getLogger(__name__)

from argparse import ArgumentParser
from json import loads
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

parser = ArgumentParser()
parser.add_argument("--data", type=str, default="data/timer_data.json")
parser.add_argument("--simtime", type=float, default=1.0)
parser.add_argument("--verbosity", type=int, default=20)
args = parser.parse_args()

PLOT_PARAMS = {
    "ax_width": 6,
    "ax_height": 2,
    "fig_y_margin": 1,
    "fig_x_margin": 1,
    "title_font_size": 30,
    "label_font_size": 25,
    "legend_font_size": 25,
    "tick_font_size": 20,
    "ax_y_label": "",  # MPI rank is prepended to this string
    "ax_x_label": "Spike times [ms]",
    # Axes can be drawn in a single column or in individual plots
    "plot_individual": False,
    # If axes are drawn individually then each plot has a title
    # If axes are drawn together, only top level axe will have a title
    # If axes are drawn together only bottom level axe will have x label
    # If axes are drawn individually one file per axe will be created
    # in that case the rank of the process will be used as suffix for the file name
    "file_format": "png",  # Plot file format
}

TIMER_MAP = {
    "time_init": "Initialization time",
    "time_create": "Node creation time",
    "time_connect": "Node connection time",
    "time_calibrate": "Calibration time",
    "time_presim": "Pre-simulation time",
    "time_sim": "Simulation time",
    "time_gather": "Data gathering time",
    "net_con": "Network construction time",
    "rtf": "Real-Time factor",
}


def update_logging_level() -> None:
    """
    Function to initialize Python logging.
    """
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(levelname)s:\n%(message)s"))
    logging.basicConfig(level=args.verbosity, handlers=(stdout,))


def load_and_flatten_data():
    """
    Function to flatten timer data collected across simulations
    Structure of data is expected as:
    dictionary(
        configuration string: dictionary(
            "n-t-s": tuple(
                number of nodes used as string,
                number of tasks used as string,
                scale used as string,
            )
            "timers": list(
                simulation run timer data as list(
                    rank timer data as dictionary(
                        "rank": MPI rank id
                        "timer_*": timer data
                    )
                    ...
                )
                ...
            )
        )
        ...
    )
    """
    data: dict = loads(Path(args.data).read_text())
    flat_data = []
    keys = ["nodes", "tasks", "scale"]
    append_timer_keys = True

    for n_t_s in data.values():
        nodes = int(n_t_s["n-t-s"][0])
        assert 1 <= nodes <= 2
        tasks = int(n_t_s["n-t-s"][1])
        assert 1 <= tasks <= 8
        scale = int(n_t_s["n-t-s"][2])
        assert scale in (1, 10, 100, 1000, 10000, 100000)
        timer_data: list = n_t_s["timers"]
        assert len(timer_data) == 10
        for run_timers in timer_data:
            for rank_timers in run_timers:
                if append_timer_keys:
                    keys.extend(rank_timers.keys())
                    # We also append calculated keys
                    # for the network construction time and real time factor
                    keys.append("net_con")
                    keys.append("rtf")
                    append_timer_keys = False
                flat_data.append(
                    (
                        nodes,
                        tasks,
                        scale,
                        *rank_timers.values(),
                        # The network construction time is the sum of
                        # time to create neurons, time to connect, and time to calibrate
                        rank_timers["time_create"]
                        + rank_timers["time_connect"]
                        + rank_timers["time_calibrate"],
                        # The real time factor is the ratio of
                        # the time to simulate and the total simulation time
                        rank_timers["time_sim"] / args.simtime,
                    )
                )

    return pd.DataFrame(data=flat_data, columns=keys)


if __name__ == "__main__":
    update_logging_level()
    flat_data = load_and_flatten_data()
    LOG.info(flat_data.head(3))

    # Create a figure per timer
    for timer in TIMER_MAP:
        assert timer in flat_data.columns
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(
                3 * PLOT_PARAMS["fig_x_margin"] + 2 * PLOT_PARAMS["ax_width"],
                2 * PLOT_PARAMS["fig_y_margin"] + PLOT_PARAMS["ax_height"],
            ),
        )
        for ax in axs:
            ax.set_title(TIMER_MAP[timer], fontsize=PLOT_PARAMS["title_font_size"])
            ax.set_xlabel("Network scale", fontsize=PLOT_PARAMS["label_font_size"])
            ax.tick_params(labelsize=PLOT_PARAMS["tick_font_size"])

        if timer == "rtf":
            axs[0].set_ylabel(
                r"$\mathit{T}_{ wall }\;/\;\mathit{T}_{ model }$",
                fontsize=PLOT_PARAMS["label_font_size"],
            )
            axs[1].set_ylabel("Log RTF", fontsize=PLOT_PARAMS["label_font_size"])
        else:
            axs[0].set_ylabel("Time [ms]", fontsize=PLOT_PARAMS["label_font_size"])
            axs[1].set_ylabel("Log time", fontsize=PLOT_PARAMS["label_font_size"])

        for idx in (0, 1):
            # See https://seaborn.pydata.org/generated/seaborn.boxenplot.html#seaborn.boxenplot
            sns.boxenplot(
                data=flat_data,
                x="scale",
                y=timer,
                hue="tasks",
                fill=True,
                gap=0.1,
                palette="pastel",
                ax=axs[idx],
                log_scale=idx == 1,
            )
        plot_file = Path(
            f"leonardo_ring_benchmark_{timer}.{PLOT_PARAMS['file_format']}"
        )
        fig.tight_layout()
        fig.savefig(plot_file, dpi=300)
