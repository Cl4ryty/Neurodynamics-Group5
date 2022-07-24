import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self, ns=10):
        super(Model, self).__init__()
        # initialize layers
        self.layer_list = [tf.keras.layers.Dense(units=ns, activation=tf.nn.silu,
                                                 kernel_initializer=tf.random_normal_initializer(),
                                                 bias_initializer=tf.random_normal_initializer()),
                           tf.keras.layers.Dense(units=ns, activation=tf.nn.sigmoid,
                                                 kernel_initializer=tf.random_normal_initializer(),
                                                 bias_initializer=tf.random_normal_initializer()),
                           tf.keras.layers.Dense(units=1, activation=tf.nn.selu,
                                                 kernel_initializer=tf.random_normal_initializer(),
                                                 bias_initializer=tf.random_normal_initializer())

                           ]

    @tf.function
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x


def training_step(model, input, loss_function, optimizer):
    """
    Performs a training step of the model using the given input,
    calculating the loss with the given function and then using the optimizer to optimize the model.

    :param tf.keras.Model model: the model to be trained
    :param tf.Tensor input: the input
    :param tf.keras.losses.Loss loss_function: the loss function
    :param tf.keras.optimizers.Optimizer optimizer: the optimizer
    :return: loss - the loss for this training step
    :rtype: tf.Tensor
    """
    with tf.GradientTape() as tape:
        f = model(input)
        loss = loss_function(f, input)
        gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss




