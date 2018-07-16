from typing import Tuple

import os
import shutil

import numpy as np
import xarray as xr
import yaml

import pkg_constants
from api.models.nb_glm import Simulator, Estimator

import utils.stats as stat_utils


def init_benchmark(
        root_dir: str,
        sim: Simulator,
        batch_size,
        stop_at_step=5000,
        learning_rate=0.05,
        save_checkpoint_steps=25,
        save_summaries_steps=25,
        export_steps=25,
        config_file="config.yml"
):
    os.makedirs(root_dir, exist_ok=True)

    config = {
        "sim_data": "sim_data.h5",
        "plot_dir": "plot_dir",
    }

    os.makedirs(os.path.join(root_dir, config["plot_dir"]), exist_ok=True)
    sim.save(os.path.join(root_dir, config["sim_data"]))

    config["benchmark_samples"] = {
        "minibatch_sgd": prepare_benchmark_sample(
            root_dir=root_dir,
            working_dir="minibatch_sgd",
            batch_size=batch_size,
            stop_at_step=stop_at_step,
            learning_rate=learning_rate,
            save_checkpoint_steps=save_checkpoint_steps,
            save_summaries_steps=save_summaries_steps,
            export_steps=export_steps,
        ),
        "gradient_descent": prepare_benchmark_sample(
            root_dir=root_dir,
            working_dir="gradient_descent",
            batch_size=sim.num_observations,
            stop_at_step=stop_at_step,
            learning_rate=learning_rate,
            save_checkpoint_steps=save_checkpoint_steps,
            save_summaries_steps=save_summaries_steps,
            export_steps=export_steps,
        )
    }

    config_file = os.path.join(root_dir, config_file)
    with open(config_file, mode="w") as f:
        yaml.dump(config, f, default_flow_style=False)


def prepare_benchmark_sample(
        root_dir: str,
        working_dir: str,
        batch_size: int,
        stop_at_step: int,
        learning_rate: float = 0.05,
        stop_below_loss_change: float = None,
        save_checkpoint_steps=25,
        save_summaries_steps=25,
        export_steps=25,
        **kwargs

):
    os.makedirs(os.path.join(root_dir, working_dir), exist_ok=True)

    sample_config = {
        "working_dir": working_dir,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }

    init_args = {
        # "working_dir": working_dir,
        "stop_at_step": stop_at_step,
        "stop_below_loss_change": stop_below_loss_change,
        "save_checkpoint_steps": save_checkpoint_steps,
        "save_summaries_steps": save_summaries_steps,
        "export_steps": export_steps,
        "export": ["a", "b", "mu", "r", "loss"],
        **kwargs
    }

    sample_config["init_args"] = init_args

    return sample_config


def get_benchmark_samples(root_dir: str, config_file="config.yml"):
    config_file = os.path.join(root_dir, config_file)
    with open(config_file, mode="r") as f:
        config = yaml.load(f)
    return list(config["benchmark_samples"].keys())


def run_benchmark(root_dir: str, sample: str, config_file="config.yml"):
    config_file = os.path.join(root_dir, config_file)
    with open(config_file, mode="r") as f:
        config = yaml.load(f)

    sim_data_file = os.path.join(root_dir, config["sim_data"])

    sample_config = config["benchmark_samples"][sample]

    working_dir = os.path.join(root_dir, sample_config["working_dir"])
    batch_size = sample_config["batch_size"]
    learning_rate = sample_config["learning_rate"]

    init_args = sample_config["init_args"]
    init_args["working_dir"] = working_dir

    print("loading data...", end="", flush=True)
    sim = Simulator()
    sim.load(sim_data_file)
    print("\t[OK]")

    print("starting estimation of benchmark sample '%s'..." % sample)
    estimator = Estimator(sim.input_data, batch_size=batch_size)
    estimator.initialize(**init_args)
    estimator.train(learning_rate=learning_rate)
    print("estimation of benchmark sample '%s' ready" % sample)


def load_benchmark_dataset(root_dir: str, config_file="config.yml") -> Tuple[Simulator, xr.Dataset]:
    config_file = os.path.join(root_dir, config_file)
    with open(config_file, mode="r") as f:
        config = yaml.load(f)

    sim_data_file = os.path.join(root_dir, config["sim_data"])
    sim = Simulator()
    sim.load(sim_data_file)

    benchmark_samples = config["benchmark_samples"]
    benchmark_data = []
    for smpl, cfg in benchmark_samples.items():
        data = xr.open_mfdataset(
            os.path.join(root_dir, cfg["working_dir"], "estimation-*.h5"),
            engine="netcdf4",
            concat_dim="step",
            autoclose=True,
            parallel=True,
        )
        data = data.sortby("global_step")
        data.coords["benchmark"] = smpl
        benchmark_data.append(data)
    benchmark_data = xr.auto_combine(benchmark_data, concat_dim="benchmark", coords="all")

    return sim, benchmark_data


def plot_benchmark(root_dir: str, config_file="config.yml"):
    print("loading config...", end="", flush=True)
    config_file = os.path.join(root_dir, config_file)
    with open(config_file, mode="r") as f:
        config = yaml.load(f)
    print("\t[OK]")

    plot_dir = os.path.join(root_dir, config["plot_dir"])

    print("loading data...", end="", flush=True)
    sim, benchmark_data = load_benchmark_dataset(root_dir)
    benchmark_data.coords["time_elapsed"] = benchmark_data.time_elapsed.cumsum("step")
    print("\t[OK]")

    import plotnine as pn
    import matplotlib.pyplot as plt

    from dask.diagnostics import ProgressBar

    def plot_stat(val, val_name, name_prefix, scale_y_log10=False):
        with ProgressBar():
            df = val.to_dataframe(val_name).reset_index()

        plot = (pn.ggplot(df)
                + pn.aes(x="time_elapsed", y=val_name, group="benchmark", color="benchmark")
                + pn.geom_line()
                + pn.geom_vline(xintercept=df.loc[[np.argmin(df[val_name])]].time_elapsed.values[0], color="black")
                + pn.geom_hline(yintercept=np.min(df[val_name]), alpha=0.5)
                )
        if scale_y_log10:
            plot = plot + pn.scale_y_log10()
        plot.save(os.path.join(plot_dir, name_prefix + ".time.svg"), format="svg")

        plot = (pn.ggplot(df)
                + pn.aes(x="global_step", y=val_name, group="benchmark", color="benchmark")
                + pn.geom_line()
                + pn.geom_vline(xintercept=df.loc[[np.argmin(df[val_name])]].global_step.values[0], color="black")
                + pn.geom_hline(yintercept=np.min(df[val_name]), alpha=0.5)
                )
        if scale_y_log10:
            plot = plot + pn.scale_y_log10()
        plot.save(os.path.join(plot_dir, name_prefix + ".step.svg"), format="svg")

    print("plotting...")
    val: xr.DataArray = stat_utils.rmsd(
        np.exp(xr.DataArray(sim.params["a"][0], dims=("features",))),
        np.exp(benchmark_data.a.isel(design_loc_params=0)), axis=[0])
    plot_stat(val, "mapd", "real_mu")

    val: xr.DataArray = stat_utils.rmsd(
        np.exp(xr.DataArray(sim.params["b"][0], dims=("features",))),
        np.exp(benchmark_data.b.isel(design_scale_params=0)), axis=[0])
    plot_stat(val, "mapd", "real_r")

    val: xr.DataArray = benchmark_data.loss
    plot_stat(val, "loss", "loss")

    print("ready")


def clean(root_dir: str):
    for the_file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', help='root directory of the benchmark', required=True)
    subparsers = parser.add_subparsers(help='select an action')

    act_init = subparsers.add_parser('init', help='set up a benchmark')
    act_init.set_defaults(action='init')
    act_init.add_argument('--num_observations', help='number of observations to generate', type=int, default=4000)
    act_init.add_argument('--num_features', help='number of features to generate', type=int, default=500)
    act_init.add_argument('--batch_size', help='batch size to use for mini-batch SGD', type=int, default=500)
    act_init.add_argument('--learning_rate', help='learning rate to use for all optimizers', type=float, default=0.05)
    act_init.add_argument('--stop_at_step', help='number of steps to run', type=int, default=5000)
    act_init.add_argument('--save_checkpoint_steps', help='number of steps to run', type=int, default=25)
    act_init.add_argument('--save_summaries_steps', help='number of steps to run', type=int, default=25)
    act_init.add_argument('--export_steps', help='number of steps to run', type=int, default=25)

    act_run = subparsers.add_parser('run', help='run a benchmark')
    act_run.set_defaults(action='run')
    act_run.add_argument('--benchmark_sample', help='If specified, only this benchmark sample will be executed')

    act_print_samples = subparsers.add_parser('print_samples', help='print all benchmark samples')
    act_print_samples.set_defaults(action='print_samples')

    act_plot = subparsers.add_parser('plot', help='generate plots')
    act_plot.set_defaults(action='plot')

    act_clean = subparsers.add_parser('clean', help='clean up root directory')
    act_clean.set_defaults(action='clean')

    args, unknown = parser.parse_known_args()

    root_dir = os.path.expanduser(args.root_dir)

    action = args.action
    if action == "init":
        sim = Simulator(num_observations=args.num_observations, num_features=args.num_features)
        sim.generate()

        init_benchmark(
            root_dir=root_dir,
            sim=sim,
            batch_size=args.batch_size,
            stop_at_step=args.stop_at_step,
            learning_rate=args.learning_rate,
            save_checkpoint_steps=args.save_checkpoint_steps if args.save_checkpoint_steps > 0 else None,
            save_summaries_steps=args.save_summaries_steps if args.save_summaries_steps > 0 else None,
            export_steps=args.export_steps if args.export_steps > 0 else None,
        )
    elif action == "run":
        if args.benchmark_sample is not None:
            run_benchmark(root_dir, args.benchmark_sample)
        else:
            benchmark_samples = get_benchmark_samples(root_dir)
            for smpl in benchmark_samples:
                run_benchmark(root_dir, smpl)
    elif action == "print_samples":
        benchmark_samples = get_benchmark_samples(root_dir)
        for smpl in benchmark_samples:
            print(smpl)
    elif action == "plot":
        plot_benchmark(root_dir)
    elif action == "clean":
        clean(root_dir)


if __name__ == '__main__':
    main()
