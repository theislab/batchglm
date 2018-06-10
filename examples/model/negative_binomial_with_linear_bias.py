import os
import datetime

from models.negative_binomial_linear_biased import Simulator
from models.negative_binomial_linear_biased.estimator import Estimator

from examples.util import stat_frame
from examples.model import simulate
from utils.config import getConfig


def estimate(sim: Simulator, wd):
    estimator = Estimator(sim.data, batch_size=500)
    # estimator.validate_data()
    estimator.initialize(
        working_dir=wd,
        save_checkpoint_steps=25,
        save_summaries_steps=25,
        export=["a", "b", "mu", "r", "loss"],
        export_steps=25
    )
    
    return estimator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs=1, help='config file')
    parser.add_argument('--data', nargs=1, help='folder for sample data')
    parser.add_argument('--generate', nargs="?", default=False, const=True,
                        help="generate new sample data; if not specified, existing data is assumed in the data folder")
    args, unknown = parser.parse_known_args()
    
    config = getConfig(args.config)
    
    data_folder = args.data
    if data_folder is None:
        data_folder = config.get("DEFAULT", "data_folder", fallback="data/")
    
    generate_new_data = args.generate
    
    sim_data_folder = os.path.join(data_folder, Simulator.__module__)
    
    sim = Simulator()
    simulate(sim, data_folder=sim_data_folder, generate_new_data=generate_new_data)
    
    estimator = estimate(sim, wd=os.path.join(data_folder,
                                              "log",
                                              Estimator.__module__,
                                              datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    sim.save(os.path.join(estimator.working_dir, "sim_data.h5"))
    
    print("loss: %f" % estimator.loss)
    stats = stat_frame(estimator, sim, ["mu", "sigma2"])
    print(stats)
    
    estimator.train(learning_rate=0.05)
    
    print("loss: %f" % estimator.loss)
    stats = stat_frame(estimator, sim, ["mu", "sigma2"])
    print(stats)
