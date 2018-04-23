from models.negative_binomial_mixture_linear_biased import Simulator
from models.negative_binomial_mixture_linear_biased.estimator import Estimator

from examples.util import stat_frame
import utils.stats as stat_utils


def simulate(data_folder=None, generate_new_data=False):
    sim = Simulator()

    if generate_new_data:
        print("Generating new data...")
        sim.generate()
        if data_folder is not None:
            print("Saving data...")
            sim.save(data_folder)
    elif data_folder is not None:
        print("Loading data...")
        sim.load(data_folder)
    else:  # no arguments specified
        print("Generating new data...")
        sim.generate()

    return sim


def estimate(sim: Simulator):
    estimator = Estimator(sim.data)
    estimator.validate_data()
    estimator.initialize()
    # estimator.train(steps=100)

    return estimator


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs=1, help='folder for sample data')
    parser.add_argument('--generate', nargs="?", default=False, const=True,
                        help="generate new sample data; if not specified, existing data is assumed in the data folder")
    args, unknown = parser.parse_known_args()

    data_folder = args.data
    generate_new_data = args.generate

    sim = simulate(data_folder, generate_new_data)
    estimator = estimate(sim)

    print("loss: %d" % estimator.loss)
    stats = stat_frame(estimator, sim, ["mu", "sigma2"])
    print(stats)
    print("MAE of mixture probs: %.4f" % stat_utils.mae(estimator.mixture_prob, sim.mixture_prob))

    for i in range(10):
        estimator.train(steps=10, learning_rate=0.05)

        print("loss: %d" % estimator.loss)
        stats = stat_frame(estimator, sim, ["mu", "sigma2"])
        print(stats)
        print("MAE of mixture probs: %.4f" % stat_utils.mae(estimator.mixture_prob, sim.mixture_prob))


