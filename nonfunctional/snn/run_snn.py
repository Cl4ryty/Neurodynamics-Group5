import os
import re
import numpy as np
from matplotlib import pyplot as plt
import pyNN.spiNNaker as sim

sim.setup(timestep=1.0, min_delay=1.0)

# load /build model for each de

# TODO save the files for each de in a specific folder
# navigate to the respective folder

path = "snn_test"
print(os.getcwd())
# os.chdir(path)
files = sorted(os.listdir(path))
print("layer files:", files)

layers = []

celltype = sim.SpikeSourceArray()

# add input layer
layers.append(sim.Population(
    1, celltype,
    label='InputLayer'))

cell_params = {
    'cm': 0.25, 'tau_m': 10.0, 'tau_refrac': 2.0,
    'tau_syn_E': 2.5, 'tau_syn_I': 2.5,
    'v_reset': -70.0, 'v_rest': -65.0, 'v_thresh': -55.0}


# set up the network
# for each layer -> there are two files per layer, so just take num files/2
for i in range(len(files)//2):

    # create next layer
    if i < len(files) - 2:
        # get number of neurons from file -> number between _ _
        result = re.search('_(.*)_', files[i*2])
        ns = int(result.group(1))
        print(ns, "ns in layer", i + 2 / 2)
    # else:
    #     # use the number of neurons of this layer
    #     result = re.search('_(.*)_', files[i])
    #     ns = int(result.group(1))
    #     print(ns, "ns in layer", i / 2)

    layer2 = sim.Population(ns, sim.IF_cond_exp(**cell_params))

    if i == 0:
        # create first layer
        # get number of neurons from file -> number between _ _
        result = re.search('_(.*)_', files[i])
        ns = int(result.group(1))
        print(ns, "ns in layer", i / 2)
        layer1 = sim.Population(ns, sim.IF_cond_exp(**cell_params))
        layers.append(layer1)
    else:
        # current layer is already in layer list
        layer1 = layers[-1]

    layers.append(layer2)

    # create projections for exitatory an inhibitory connections
    print("connection from layer", i, "to", i+1, files[2*i], files[2*i+1])
    sim.Projection(layer1, layer2, sim.FromFileConnector(os.path.join(path,files[2*i])))
    sim.Projection(layer1, layer2, sim.FromFileConnector(os.path.join(path,files[2*i+1])))



# configure recording of the outputs
layers[-1].record('spikes')

# set the inputs
sim_duration = 10.0 # seconds
num_inputs = 1
inputs = np.arange(num_inputs, dtype=int)

results = []

for input in inputs:


    np.linspace(0, int(sim_duration), int(sim_duration) * 1)


    spike_times = [np.linspace(0, int(sim_duration), int(sim_duration) * amplitude) for amplitude in [input]]

    layers[0].set(spike_times=spike_times)

    # run the simulation
    sim.run(sim_duration)



    # retrieve recorded data
    #
    # def get_spiketrains_input(self):
    #     shape = list(self.parsed_model.input_shape) + [self._num_timesteps]
    #     spiketrains_flat = self.layers[0].get_data(
    #         'spikes').segments[-1].spiketrains
    #     spiketrains_b_l_t = self.reshape_flattened_spiketrains(
    #         spiketrains_flat, shape)
    #     return spiketrains_b_l_t


    spike_counts = layers[-1].get_spike_counts()
    print("spike_counts", spike_counts)
    output_firing_rates = np.array(
        [value for (key, value) in sorted(spike_counts.items())])/sim_duration

    results.append(output_firing_rates)

results = np.array(results)
results = results.flatten()

# plot graph
plt.plot(np.arange(0,len(results)), results)
plt.xlabel("x?")
plt.ylabel("Output firing rate (spikes/second)")
plt.show()
plt.savefig("simple_example.png")
