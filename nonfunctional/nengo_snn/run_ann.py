import tensorflow as tf
from matplotlib import pyplot as plt

from ann import Model1, loss_func2, training_step, loss_func1, training_step2

# we want to solve the ode in this range
inputs = tf.linspace(-2., 2., num=401)

# inputs = tf.linspace(0., 1., num=100)
inputs = tf.expand_dims(inputs, -1)

# set hyperparameters
epochs = 2000
learning_rate = 0.01

# initialize model, loss and optimizer
model = Model1(10)


def loss_func1_mod(model, y_true):
    x = y_true
    # f = tf.reshape(y_pred, x.shape)

    # print("y_true", y_true)
    # print("y pred", y_pred)
    # print()
    # print()
    with tf.GradientTape() as tape:
        tape.watch(x)
        f = model(x)
        # f = tf.squeeze(f)
        # print("f", f)
        # Add print operation
        # tf.print("f: ", [f])
        # print("y_pred as f", tf.reshape(y_pred, x.shape))
        # tf.print("y_pred as f:", tf.reshape(y_pred, x.shape))
        df_dx = tape.gradient(f, x)
        # print("dfdx", df_dx)

        # y pred is the same as f so just use it to get the correct gradient in the future
        # f = tf.reshape(y_pred, x.shape)
        eq = df_dx + 2. * x * f
        # print("eq", eq)
        ic = model(tf.constant(0.0, shape=(1, 1))) - 1.
        # print("ic", ic)
    result = tf.math.reduce_mean(tf.square(eq)) + tf.square(ic)
    # print("loss returned:", result)
    return result


def loss_func1_mod2(y_pred, y_true):
    x = y_true
    with tf.GradientTape() as tape:
        tape.watch(x)
        f = model(x)
        df_dx = tape.gradient(f, x)
        eq = df_dx + 2. * x * f
        ic = model(tf.constant(0.0, shape=(1, 1))) - 1.
    return tf.math.reduce_mean(tf.square(eq)) + tf.square(ic)


loss_function = loss_func1_mod2
optimizer = tf.keras.optimizers.Adam(learning_rate)
# Initialize lists for later visualization.
train_losses = []
test_losses = []
test_accuracies = []

# We train for epochs.
for epoch in range(epochs):
    # training (and checking in with training)
    epoch_loss_agg = []

    if epoch % 100 == 0:
        f = model(inputs)
        print(f'Epoch: {str(epoch)} starting with loss {loss_function(f, inputs)}')
    train_loss = training_step2(model, inputs, loss_function, optimizer)
    train_losses.append(tf.squeeze(train_loss))

print(train_losses[0])
# Visualize accuracy and loss for training and test data.
plt.figure()
plt.plot(train_losses)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.show()

approx = model(inputs)
plt.plot(tf.squeeze(inputs), tf.squeeze(approx))
plt.show()
