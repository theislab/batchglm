import os

def simulate(sim, data_folder=None, generate_new_data=False):
    if generate_new_data:
        print("Generating new data...")
        sim.generate()
        if data_folder is not None:
            print("Saving data...")
            sim.save(data_folder)
    elif data_folder is not None:
        if os.path.exists(data_folder):
            print("Loading data...")
            sim.load(data_folder)
        else:
            print("Generating new data...")
            os.mkdir(data_folder)
            sim.generate()
            sim.save(data_folder)
    else:  # no arguments specified
        print("Generating new data...")
        sim.generate()
    
    return sim
