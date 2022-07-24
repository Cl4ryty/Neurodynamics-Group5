import matplotlib.pyplot as plt
import os
import tensorflow as tf

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

# get paths to all models
root = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'temp', "1656317105.6027677")
dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) and not (item == "plots" or item=="log")]
print(dirlist)

for model_path in dirlist:

    complete_path = os.path.join(root, model_path)

    # SNN TOOLBOX CONFIGURATION #
    #############################

    # Create a config file with experimental setup for SNN Toolbox.
    configparser = import_configparser()
    config = configparser.ConfigParser()

    config['paths'] = {
        'path_wd': root,  # Path to model.
        'dataset_path': root,  # Path to dataset.
        'filename_ann': model_path  # Name of input model.
    }

    config['tools'] = {
        'evaluate_ann': True,  # Test ANN on dataset before conversion.
        'normalize': False  # Normalize weights for full dynamic range.
    }

    config['simulation'] = {
        'simulator': 'INI',  # Chooses execution backend of SNN toolbox.
        'duration': 50,  # Number of time steps to run each sample.
        'num_to_test': 400,  # How many test samples to run.
        'batch_size': 400,  # Batch size for simulation.
        'keras_backend': 'tensorflow'  # Which keras backend to use.
    }

    config['input'] = {
        'model_lib': 'keras'  # Input model is defined in pytorch.
    }

    config['output'] = {
        'plot_vars': {  # Various plots (slows down simulation).
            'spiketrains',  # Leave section empty to turn off plots.
            'spikerates',
            'activations',
            'correlation',
            'v_mem',
            'error_t'}
    }

    # Store config file.
    config_filepath = 'config'
    with open(config_filepath, 'w') as configfile:
        config.write(configfile)

    # Need to copy model definition over to ``path_wd`` (needs to be in same dir as
    # the weights saved above).
    # source_path = inspect.getfile(Model)
    # shutil.copyfile(source_path, os.path.join(path_wd, model_name + '.py'))

    # RUN SNN TOOLBOX #
    ###################

    main(config_filepath)

    break