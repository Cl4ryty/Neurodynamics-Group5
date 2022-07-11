# Neurodynamics-Group5
# recent updates /remarks
- Link to (WIP) ANN training notebook on colab: https://colab.research.google.com/drive/1NwvXcDmwGfrzuEHoj2G883BXYrahjgft?usp=sharing

## File structure / remarks
- the code to train the ANN, store the results, and run the conversion with SNN toolbox is in `run_ann_training.py"
- results are in `temp/`, each subfolder is one run, ideally the plots are in a subfolder of those called plots (not the case for all runs, as it was not implemented for the first few runs), collected metrics are stored in `metrics.txt`, hyperparameters used in `hyperparameters.txt`, and activations used for the NN layers in `activations.txt`. Not all files might be available for the first runs.
- order of collected metrics in `metrics.txt`: de_names, final_losses, final_errors, first_epoch_under_threshold, time_to_threshold, total_training_time

## How to use
- clone this repository
- Install conda (e.g. https://www.anaconda.com/) if you don't have already

### Running the ANN (and conversion) / adding DEs:
  - create a new environment with a working python version and pip running `conda create -n env_name python==3.9.12 pip`, substituting env_name with a name of your liking
  - activate the environment `conda activate env_name`
  - navigate into the cloned repo
  - install the required libraries with `pip install -r requirements.txt`
  - finish setting up the spinnaker library by running `python -m spynnaker8.setup_pynn`
  - you might have to edit the `.spynnaker.cfg` file that should be located in your home directory (if it does not exist yet it will be created after running the code once) – setting `virtual_board = True` and setting the `width` and `height` to allowed values (e.g. 2) – see https://spinnakermanchester.github.io/spynnaker/5.0.0/PyNNOnSpinnakerInstall.html#Configuration for more information
  
### Testing the SNN toolbox:
  - create a new environment and install pip in it with `conda create -n env_name pip`, substituting env_name with a name of your liking
  - activate the environment `conda activate env_name`
  - navigate into the LibraryTests folder of the cloned repo
  - install the required libraries with `pip install -r requirements.txt`
  - run the `toolbox_test` to train the ann and convert it with the snn toolbox -> `python LibraryTests/toolbox_test.py`


## Implementiing DEs for the ANN to solve
`run_ann_des.py` contains the class for simple implementation of DEs as well as some already implemented DEs as examples and the (not completely finished) code to train the ANN to solve the equations.
To add a DE create a new DE object and append it to the equations list. The code contains examples for creating DEs as well as comments about the parameters.
Things to note:
- only operations that can be applied to tensors should be used in the equation and solution functions (`+`, `-`, `*`, and `/` are fine to use, but for other operations use the tf version)
- the equation eq should have the parameters `df_dx`, (`df_dxx`, `df_d_xxx`, `df_d_xxxx`), `f`, `x`  – with df_dx, f, and x being required for all DEs and further derivatives only being required for the higher order DEs that use them. A third order DE would then have the parameters `df_dx, df_dxx, df_d_xxx, f, x`


## Useful links
- Nengo tutorial for converting ann to snn: https://www.nengo.ai/nengo-dl/v3.5.0/examples/keras-to-snn.html
