# Paper Title
TODO: short project description, link to paper in repo.
For further information, please refer to our [paper](../blob/master/paper).

## Repository structure
- `Analysis/` contains contains csv files of some of the processed data as well as an Rmd notebook to generate the plots used in the paper and calculate some statistics
- `utils/` contains some utility scripts, notably `save_for_analysis.py` which can be used to save the data collected during the ANN training as a csv file – [time_metrics.csv in Analysis](../main/Analysis/time_metrics.csv) – to be more easily used for analysis
- `run_ann_training.py` contains the code to train the ANN, and store the results
- results are in `data/`, each timestamped subfolder is one complete training run, which contains
  - numbered subfolders for all subruns over which the time metrics are calculated, which contain:
    - the saved trained ANNs as folders named after the DEs they were trained to solve
    - `plots/`: folder containig plots of the training loss, training error, and model solution vs. true solution for all ANNs/DEs
    - `activations.txt`: the activation functions used in the ANNs
    - `metrics.txt`: metrics for all DEs to be used for later analysis
    - `serialized_custom_loss_functions.txt`: seriazed dictionary of the custom loss function used for trainine each ANN with the DE names as keys – needed for loading the saved ANNs
  - `hyperparameters.txt`: the hyperparamete values used for this training run
  - `system_info.txt`: a file conatining information about the system the training was run on – only when the training was done with the jupyter notebook
  - `run_metrics.npy`: serialized numpy array of all the metrics collected for all subruns 
  - `plots/`: folder containing plots of the time metrics calculated over all subruns
  
## How to use
- clone this repository
- Install conda (e.g. https://www.anaconda.com/) if you don't have already

### Running the ANN training:
  - use the [colab notebook](https://colab.research.google.com/drive/1NwvXcDmwGfrzuEHoj2G883BXYrahjgft?usp=sharing)
  - run it locally:
    - create a new environment with and already install pip by running `conda create -n env_name pip`, substituting env_name with a name of your liking
    - activate the environment `conda activate env_name`
    - navigate into the cloned repo
    - install the required libraries with `pip install -r ann_requirements.txt`
    - run the code with `python run_ann_training.py`


### Implementing DEs for the ANN to solve
`run_ann_des.py` contains the class for simple implementation of DEs as well as some already implemented DEs as examples and the (not completely finished) code to train the ANN to solve the equations.
To add a DE create a new DE object and append it to the equations list. The code contains examples for creating DEs as well as comments about the parameters.
Things to note:
- only operations that can be applied to tensors should be used in the equation and solution functions (`+`, `-`, `*`, and `/` are fine to use, but for other operations use the tf version)
- the equation eq should have the parameters `df_dx`, (`df_dxx`, `df_d_xxx`, `df_d_xxxx`), `f`, `x`  – with df_dx, f, and x being required for all DEs and further derivatives only being required for the higher order DEs that use them. A third order DE would then have the parameters `df_dx, df_dxx, df_d_xxx, f, x`
