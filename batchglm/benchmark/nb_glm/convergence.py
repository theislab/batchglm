from typing import Tuple

import os
import shutil

import scipy.stats
import numpy as np
import xarray as xr
import yaml

from .base import init_benchmark, get_benchmark_samples, run_benchmark, Simulator

import batchglm.utils.stats as stat_utils


def load_benchmark_dataset(root_dir: str, config_file="config.yml") -> Tuple[Simulator, xr.Dataset, dict]:
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

    return sim, benchmark_data, benchmark_samples


def plot_benchmark(root_dir: str, config_file="config.yml"):
    print("loading config...", end="", flush=True)
    config_file = os.path.join(root_dir, config_file)
    with open(config_file, mode="r") as f:
        config = yaml.load(f)
    print("\t[OK]")

    plot_dir = os.path.join(root_dir, config["plot_dir"])

    print("loading data...", end="", flush=True)
    sim, benchmark_data, benchmark_sample_config = load_benchmark_dataset(root_dir)
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
                + pn.geom_vline(xintercept=df.location[[np.argmin(df[val_name])]].time_elapsed.values[0], color="black")
                + pn.geom_hline(yintercept=np.min(df[val_name]), alpha=0.5)
                )
        if scale_y_log10:
            plot = plot + pn.scale_y_log10()
        plot.save(os.path.join(plot_dir, name_prefix + ".time.svg"), format="svg")

        plot = (pn.ggplot(df)
                + pn.aes(x="global_step", y=val_name, group="benchmark", color="benchmark")
                + pn.geom_line()
                + pn.geom_vline(xintercept=df.location[[np.argmin(df[val_name])]].global_step.values[0], color="black")
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

    def plot_pval(window_size):
        print("plotting p-value with window size: %d" % window_size)

        roll1 = benchmark_data.loss.rolling(step=window_size)
        roll2 = benchmark_data.loss.roll(step=window_size).rolling(step=window_size)
        mu1 = roll1.mean().dropna("step")
        mu2 = roll2.mean().dropna("step")
        var1 = roll1.var().dropna("step")
        var2 = roll2.var().dropna("step")
        n1 = window_size
        n2 = window_size

        t, df = stat_utils.welch(mu1, mu2, var1, var2, n1, n2)
        t = t[:, window_size:]
        df = df[:, window_size:]

        pval = t.copy()
        pval[:, :] = scipy.stats.t(df).cdf(t)
        pval.plot.line(hue="benchmark")
        plt.savefig(os.path.join(plot_dir, "pval_convergence.%dsteps.svg" % window_size), format="svg")
        # plt.show()
        plt.close()

    plot_pval(100)
    plot_pval(200)
    plot_pval(400)

    benchmark_data.full_loss.plot.line(hue="benchmark")
    plt.savefig(os.path.join(plot_dir, "full_loss.svg"), format="svg")
    plt.close()

    benchmark_data.loss.plot.line(hue="benchmark")
    plt.savefig(os.path.join(plot_dir, "batch_loss.svg"), format="svg")
    plt.close()

    def plot_loss_rolling_mean(window_size):
        print("plotting rolling mean of batch loss with window size: %d" % window_size)

        benchmark_data.loss.rolling(step=window_size).mean().plot.line(hue="benchmark")
        plt.savefig(os.path.join(plot_dir, "batch_loss_rolling_mean.%dsteps.svg" % window_size), format="svg")
        plt.close()

    plot_loss_rolling_mean(25)
    plot_loss_rolling_mean(50)
    plot_loss_rolling_mean(100)
    plot_loss_rolling_mean(200)

    with ProgressBar():
        benchmark_data.full_gradient.mean(dim="features").plot.line(hue="benchmark")
    plt.savefig(os.path.join(plot_dir, "mean_full_gradient.svg"), format="svg")
    plt.close()

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
    act_init.add_argument('--num_batches', help='number of batches to simulate', type=int, default=4)
    act_init.add_argument('--num_conditions', help='number of conditions to simulate', type=int, default=2)
    act_init.add_argument('--learning_rate', help='learning rate to use for all optimizers',
                          type=float,
                          nargs='+',
                          default=0.05)
    act_init.add_argument('--stopping_criteria', help='number of steps to run', type=int, default=5000)
    act_init.add_argument('--save_checkpoint_steps', help='number of steps to run', type=int, default=100)
    act_init.add_argument('--save_summaries_steps', help='number of steps to run', type=int, default=1)
    act_init.add_argument('--optim_algo', help='optimization algorithm',
                          type=str,
                          nargs='+',
                          default="gradient_descent")
    act_init.add_argument('--export_steps', help='number of steps to run', type=int, default=1)

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
        sim.generate_sample_description(num_batches=args.num_batches, num_conditions=args.num_conditions)
        sim.generate()

        init_benchmark(
            root_dir=root_dir,
            sim=sim,
            batch_size=args.batch_size,
            stopping_criteria=args.stopping_criteria,
            learning_rate=args.learning_rate,
            save_checkpoint_steps=args.save_checkpoint_steps if args.save_checkpoint_steps > 0 else None,
            save_summaries_steps=args.save_summaries_steps if args.save_summaries_steps > 0 else None,
            export_steps=args.export_steps if args.export_steps > 0 else None,
            optim_algo=args.optim_algo
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
