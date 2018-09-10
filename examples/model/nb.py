import os

from batchglm.api.models.nb import Simulator, Estimator

from examples import stat_frame
from examples import simulate


def estimate(sim: Simulator):
    estimator = Estimator(sim.input_data)
    # estimator.validate_data()
    estimator.initialize()
    estimator.train()

    return estimator


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs=1, help='config file')
    parser.add_argument('--data', nargs=1, help='folder for sample data')
    parser.add_argument('--generate', nargs="?", default=False, const=True,
                        help="generate new sample data; if not specified, existing data is assumed in the data folder")
    args, unknown = parser.parse_known_args()

    data_folder = args.data
    if data_folder is None:
        data_folder = "data/"

    generate_new_data = args.generate

    sim_data_folder = os.path.join(data_folder, Simulator.__module__)

    sim = Simulator()
    simulate(sim, data_folder=sim_data_folder, generate_new_data=generate_new_data)

    estimator = estimate(sim)

    print("loss: %f" % estimator.loss)
    stats = stat_frame(estimator, sim, ["mu", "sigma2"])
    print(stats)


if __name__ == '__main__':
    main()
