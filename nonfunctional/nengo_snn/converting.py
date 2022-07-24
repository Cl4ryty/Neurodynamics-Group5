import nengo_dl
import matplotlib.pyplot as plt
import tensorflow as tf

# set hyperparameters
epochs = 2000
learning_rate = 0.001
ns = 10

# creating the model
inp = tf.keras.Input(shape=(1))

d1 = tf.keras.layers.Dense(units=ns, activation="sigmoid",
                           kernel_initializer=tf.random_normal_initializer(),
                           bias_initializer=tf.random_normal_initializer(),
                           name="d1")(inp)

d2 = tf.keras.layers.Dense(units=ns, activation="sigmoid",
                           kernel_initializer=tf.random_normal_initializer(),
                           bias_initializer=tf.random_normal_initializer(),
                           name="d2")(d1)

dense = tf.keras.layers.Dense(units=1, activation="relu",
                              kernel_initializer=tf.random_normal_initializer(),
                              bias_initializer=tf.random_normal_initializer(),
                              name="out")(d2)

model = tf.keras.Model(inputs=inp, outputs=dense)

# converter = nengo_dl.Converter(model)
converter = nengo_dl.Converter(
    model,
    swap_activations={tf.nn.relu: tf.nn.relu,
                      tf.nn.sigmoid: tf.nn.sigmoid},
)

inputs = tf.linspace(-2., 2., num=400)
# expand dims 2 times as nengo expects the input to have the shape (batch_size, n_steps, node.size_out)
inputs = tf.expand_dims(inputs, -1)
inputs = tf.expand_dims(inputs, -1)

with nengo_dl.Simulator(converter.net, minibatch_size=400) as sim:

    def train_objective(outputs, targets):
        # targets will be current inputs as we do not need targets but otherwise wouldn't have access to the input values

        with tf.GradientTape() as tape:
            x = targets
            tape.watch(x)

            output = sim.keras_model([x, tf.ones((sim.minibatch_size, 1, 1))])[0]  # the index is to get the output of the probe (network output)
            # tf.ones((sim.minibatch_size, 1, 1)) is the n_step parameter (with value 1 – tf.ones), which needs to have the same dimension/shape as the input data (batch_size, n_steps, node.size_out)
            dydx = tape.gradient(output, x)

            eq = dydx + 2. * x * outputs

            prediction = sim.keras_model([tf.zeros((sim.minibatch_size, 1, 1)), tf.ones((sim.minibatch_size, 1, 1))])[0] # the index is to get the output of the probe (network output)

            ic = prediction[0][0] - 1.  # zero indices to get one value (due to how the simulation works we initally have to get a whole batch)

            return tf.math.reduce_mean(tf.square(eq)) + tf.square(ic)

    # Train the network.
    sim.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss={converter.outputs[dense]: train_objective}
    )

    # to check how the model looks
    print(sim.keras_model.summary())
    dot_img_file = 'model_converted.png'
    tf.keras.utils.plot_model(sim.keras_model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)
    dot_img_file = 'model_original.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    dot_img_file = 'tensor_graph.png'
    tf.keras.utils.plot_model(sim.tensor_graph, to_file=dot_img_file, show_shapes=True)

    sim.fit(
        {converter.inputs[inp]: inputs},
        {converter.outputs[dense]: inputs},  # targets -> loss function expects input as targets to
        epochs=epochs,
    )

    # save the parameters to file
    sim.save_params("./keras_to_snn_params")

    # freeze params
    # sim.freeze_params()

# -------------------------------------------- SIMULATE THE TRAINED NETWORK --------------------------------------------

# Setup the simulation.

    # Run the simulation.
    prediction = sim.keras_model([inputs, tf.ones((sim.minibatch_size, 1, 1))])[0]


# ------------------------------------------ PLOT THE TRAINED NETWORK RESULTS ------------------------------------------

    # Plot the network output.
    plt.figure();
    plt.xlabel('Input x');
    plt.ylabel('Network Output y');
    plt.title('Network Output vs Input (Trained)')
    plt.plot(tf.squeeze(inputs), tf.squeeze(prediction), label='y ANN')
    # plt.plot(sim.trange(), f_approx(tf.squeeze(inputs)), label='y True')   # TODO: Add solution to plot
    plt.legend()
    plt.show()
