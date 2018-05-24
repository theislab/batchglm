import os


def simulate(sim, data_folder=None, data_file="data.h5", generate_new_data=False):
    data_file = os.path.join(data_folder, data_file)
    
    if generate_new_data:
        print("Generating new data...")
        sim.generate()
        
        if data_folder is not None:
            if not os.path.isdir(data_folder):
                os.mkdir(data_folder)
            
            print("Saving data...")
            sim.save(data_file)
    elif data_folder is not None:
        if os.path.exists(data_file):
            print("Loading data...")
            sim.load(data_file)
        else:
            if not os.path.isdir(data_folder):
                os.mkdir(data_folder)
            
            print("Generating new data...")
            sim.generate()
            sim.save(data_file)
    else:  # no arguments specified
        print("Generating new data...")
        sim.generate()
    
    return sim
