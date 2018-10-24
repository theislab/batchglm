import os

from collections import OrderedDict
import itertools

import pandas as pd
import yaml

from batchglm.api.models.nb_glm import Simulator, Estimator


def init_benchmark(
        root_dir: str,
        sim: Simulator,
        config_file="config.yml",
        **kwargs
):
    os.makedirs(root_dir, exist_ok=True)

    config = {
        "sim_data": "sim_data.h5",
        "plot_dir": "plot_dir",
    }

    os.makedirs(os.path.join(root_dir, config["plot_dir"]), exist_ok=True)
    sim.save(os.path.join(root_dir, config["sim_data"]))

    benchmark_samples = dict()

    kvlist = [
        (key, val) if isinstance(val, tuple) or isinstance(val, list) else (key, [val]) for key, val in kwargs.items()
    ]
    od = OrderedDict(sorted(kvlist))
    cart = list(itertools.product(*od.values()))

    df = pd.DataFrame(cart, columns=od.keys())
    df.reset_index(inplace=True)
    df["working_dir"] = ["idx_%i" % i for i in df["index"]]

    for idx, row in df.iterrows():
        name = row["working_dir"]
        benchmark_samples[name] = prepare_benchmark_sample(
            root_dir=root_dir,
            **row.to_dict()
        )
    config["benchmark_samples"] = benchmark_samples

    config_file = os.path.join(root_dir, config_file)
    with open(config_file, mode="w") as f:
        yaml.dump(config, f, default_flow_style=False)


def prepare_benchmark_sample(
        root_dir: str,
        working_dir: str,
        batch_size: int,
        save_checkpoint_steps=25,
        save_summaries_steps=25,
        export_steps=25,
        learning_rate: float = 0.05,
        convergence_criteria="step",
        stopping_criteria=5000,
        train_mu: bool = None,
        train_r: bool = None,
        use_batching=True,
        optim_algo="gradient_descent",
        **kwargs
):
    os.makedirs(os.path.join(root_dir, working_dir), exist_ok=True)

    sample_config = {
        "working_dir": working_dir,
    }

    setup_args = {
        "batch_size": batch_size,
        "extended_summary": True,
        "init_a": "standard",
        "init_b": "standard",
    }
    sample_config["setup_args"] = setup_args

    init_args = {
        # "working_dir": working_dir,
        "save_checkpoint_steps": save_checkpoint_steps,
        "save_summaries_steps": save_summaries_steps,
        "export_steps": export_steps,
        "export": ["a", "b", "loss", "gradient", "full_loss", "full_gradient", "batch_log_probs"],
    }

    sample_config["init_args"] = init_args

    training_args = {
        "learning_rate": learning_rate,
        "convergence_criteria": convergence_criteria,
        "stopping_criteria": stopping_criteria,
        "train_mu": train_mu,
        "train_r": train_r,
        "use_batching": use_batching,
        "optim_algo": optim_algo,
    }

    sample_config["training_args"] = training_args

    return sample_config


def get_benchmark_samples(root_dir: str, config_file="config.yml"):
    config_file = os.path.join(root_dir, config_file)
    with open(config_file, mode="r") as f:
        config = yaml.load(f)
    return list(config["benchmark_samples"].keys())


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def run_benchmark(root_dir: str, sample: str, config_file="config.yml"):
    config_file = os.path.join(root_dir, config_file)
    with open(config_file, mode="r") as f:
        config = yaml.load(f)

    sim_data_file = os.path.join(root_dir, config["sim_data"])

    sample_config = config["benchmark_samples"][sample]

    working_dir = os.path.join(root_dir, sample_config["working_dir"])

    # working space locking
    if os.path.exists(os.path.join(working_dir, "ready")):
        print("benchmark sample '%s' was already estimated" % sample)
        return
    if os.path.exists(os.path.join(working_dir, "lock")):
        print((
                      "benchmark sample '%s' is locked. " +
                      "Maybe there is already another instance estimating this sample?"
              ) % sample)
        return

    print("locking dir...", end="", flush=True)
    touch(os.path.join(working_dir, "lock"))
    print("\t[OK]")

    print("loading data...", end="", flush=True)
    sim = Simulator()
    sim.load(sim_data_file)
    print("\t[OK]")

    setup_args = sample_config["setup_args"]
    setup_args["input_data"] = sim.input_data

    init_args = sample_config["init_args"]
    init_args["working_dir"] = working_dir

    training_args = sample_config["training_args"]

    print("starting estimation of benchmark sample '%s'..." % sample)
    estimator = Estimator(**setup_args)
    estimator.initialize(**init_args)
    estimator.train(**training_args)
    print("estimation of benchmark sample '%s' ready" % sample)

    print("unlocking dir and finalizing...", end="", flush=True)
    os.remove(os.path.join(working_dir, "lock"))
    touch(os.path.join(working_dir, "ready"))
    print("\t[OK]")
