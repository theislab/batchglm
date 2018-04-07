from models.negative_binomial_linear_biased import Simulator, Estimator


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
    # estimator.train(steps=10)
    
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
    
    print(estimator.loss)
