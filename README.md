# An attempt to use SNNs instead of ANNs to solve differential equations
Neural networks (NNs) have been shown (Hornik, 1991) to be able to approximate functions. In particular, Physics-Informed Neural Networks (PINNs) are used to efficiently solve partial differential equations (PDEs) non-numerically (Kharazmi et al., 2019). Although Artificial Neural Networks (ANNs) have enabled groundbreaking advances, they rely on biologically inaccurate neuron models, and training is associated with high energy consumption (Kundu et al., 2021). Spiking Neural Networks (SNNs) promise to solve these issues, because they use more biologically plausible models of neurons (Li & Furber, 2021). SNNs utilize spikes, which are discrete events that occur at distinct time points, as opposed to continuous values. Theoretically, SNNs should also be able to solve differential equations (DEs) according to similar principles as used by ANNs (Tavanaei et al., 2019), since they can be converted into each other. Nevertheless, the ability to solve DEs with SNNs has rarely been investigated. Therefore, the research objective is to investigate whether and to what extent the said DEs can be solved with SNNs instead of ANNs. First, we will train ANNs using TensorFlow, and convert them into SNNs with SNN toolbox (Rueckauer et al., 2017). After the conversion, we planned to compare the differences of root-mean-square deviation (RMSD) between ANNs and SNNs. Additionally, we will compare performance of ANN when used to solve DEs of different order and different linearity.

This code contained in this repository can be used to train ANNs to solve DEs using TensorFlow. It compares training time and final losses of ANNs when used to solve DEs of different order and different linearity. [The code to convert the trained ANNs into SNNs with SNN toolbox and run them on SpiNNaker is still nonfunctional](https://github.com/Cl4ryty/spinnaker_test).

For further information about the scientific background as well as an analysis of our results, please refer to our [paper](../main/Paper.pdf).

## Repository structure
- `Analysis/` contains contains csv files of some of the processed data as well as an Rmd (R Markdown) notebook to generate the plots used in the paper and calculate some statistics (this requires R as well as the tidyverse library to be installed to be run)
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
  - use the [colab notebook](https://colab.research.google.com/github/Cl4ryty/Neurodynamics-Group5/blob/main/Run_ANN_training.ipynb)
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
