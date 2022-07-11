import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import dill as pickle
import os
import time

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser

# WORKING DIRECTORY #
#####################

# Define path where model and output files will be stored.
# The user is responsible for cleaning up this temporary directory.
path_wd = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'temp', str(time.time())))
os.makedirs(path_wd)


# create sequential model -> not actually used as the model is set again in before the training
ns = 10
model = tf.keras.Sequential([tf.keras.layers.Dense(units=ns, activation=tf.nn.sigmoid,
                                                   kernel_initializer=tf.random_normal_initializer(),
                                                   bias_initializer=tf.random_normal_initializer(), name="first",
                                                   input_shape=(1,)),
                             tf.keras.layers.Dense(units=ns, activation=tf.nn.sigmoid,
                                                   kernel_initializer=tf.random_normal_initializer(),
                                                   bias_initializer=tf.random_normal_initializer(), name="second"),
                             tf.keras.layers.Dense(units=1, activation=tf.nn.relu,
                                                   kernel_initializer=tf.random_normal_initializer(),
                                                   bias_initializer=tf.random_normal_initializer(), name="third")

                             ])


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
        loss = loss_function(input, f)
        gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# class for simple DE implementation
class DE:
    def __init__(self, name, input_min, input_max, eq, order, ic_x, ic_y, solution):
        """
        Creates an object containing all necessary information for solving a DE with a NN and evaluating the solution.

        :param str name: The name of the equation.
        :param float input_min:
        :param float input_max:
        :param eq: lambda function specifying the equation with parameters df_dx, f, x – "lambda df_dx, f, x: …" – with df_dx
        :param int order: the order of the equation (i.e. the highest order of derivative it contains), e.g. 2 for a second order DE. Currently supports only equations of order 1 - 4
        :param list[float] ic_x: list of x values of the initial conditions, e.g. for the initial conditions f(x=0)=1 and f(2)=3 this is [0., 2.]
        :param list[float] ic_y: list of y values of the initial conditions, e.g. for the initial conditions f(x=0)=1 and f(2)=3 this is [1., 3.]
        :param solution: lambda function of the solution, containing one parameter x, i.e. "lambda x: …"
        """
        # raise error if order is out of currently supported range
        if order < 0 or order > 4:
            raise ValueError("Only equations of order 1 - 4 are supported")

        self.name = name
        self.input_min = input_min
        self.input_max = input_max
        self.eq = eq
        self.order = order
        self.ic_x = ic_x
        self.ic_y = ic_y
        self.solution = solution

    def get_inputs(self, number_points):
        inputs = tf.linspace(self.input_min, self.input_max, num=number_points)
        inputs = tf.expand_dims(inputs, -1)
        return inputs

    def analytical_solution(self, x):
        return self.solution(x)

    def get_loss_function(self):
        return self.__make_ann_loss_func()

    def __make_ann_loss_func(self):

        @tf.function
        def ann_loss_function(y_true, y_pred):
            x = y_true
            with tf.GradientTape() as tape4:
                with tf.GradientTape() as tape3:
                    with tf.GradientTape() as tape2:
                        with tf.GradientTape() as tape1:
                            tape1.watch(x)
                            tape2.watch(x)
                            tape3.watch(x)
                            tape4.watch(x)
                            f = model(x)
                            df_dx = tape1.gradient(f, x)
                            if self.order == 1:
                                eq = self.eq(df_dx, f, x)
                            if self.order > 1:
                                df_dxx = tape2.gradient(df_dx, x)
                                if self.order == 2:
                                    eq = self.eq(df_dx, df_dxx, f, x)
                            if self.order > 2:
                                df_dxxx = tape3.gradient(df_dxx, x)
                                if self.order == 3:
                                    eq = self.eq(df_dx, df_dxx, df_dxxx, f, x)
                            if self.order > 3:
                                df_dxxxx = tape4.gradient(df_dxxx, x)
                                if self.order == 4:
                                    eq = self.eq(df_dx, df_dxx, df_dxxx, df_dxxxx, f, x)

                ic = 0.0
                for ic_x, ic_y in zip(self.ic_x, self.ic_y):
                    ic += tf.square(model(tf.constant(ic_x, shape=(1, 1))) - ic_y)

            return tf.math.reduce_mean(tf.square(eq)) + ic

        return ann_loss_function


# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------

# initializing the DEs and storing them in a list
equations = []

test_de = DE(name="test_de", input_min=-2., input_max=2., eq=lambda df_dx, f, x: df_dx + 2. * x * f, order=1, ic_x=[0],
             ic_y=[1], solution=lambda x: tf.exp(-x ** 2))
equations.append(test_de)

# # Solution missing for this test
# R = 1.
# test_de1 = DE(input_min=0., input_max=1., eq=lambda df_dx, f, x: df_dx - R * x * (1 - x), order=1, ic_x=0, ic_y=1, solution=lambda x: None)
# equations.append(test_de1)


# ########################   linear DEs   #####################
# # ---------------------   first order   ---------------------

# solution might be wrong
a = 1.
b = 1.
gompertz = DE(name="gompertz", input_min=-2., input_max=2.,
              eq=lambda df_dx, f, x: f * (a - b * tf.math.log(f)),
              order=1, ic_x=[0.], ic_y=[np.e],
              solution=lambda x: tf.exp(1.))
# equations.append(gompertz)

# Kirchhoff’s law
# dürfte keinen Sinn ergeben, da zwei verschränkte Gleichungen verwendet werden: E(t) & I(t)
# pürfen, ob Ergebnis plausibel
L = 4
R = 12
E_t = 60
kirchhoff = DE(name="kirchhoff", input_min=-0.5, input_max=0.5,
               eq=lambda dI_dt, I, t: L * dI_dt + R * I - E_t,
               order=1, ic_x=[1.], ic_y=[4.75],
               solution=lambda x: 5 * (1 - tf.exp(-3 * x)))
equations.append(kirchhoff)

# Newtons first Law of cooling
# auch hier sind zu viele Bedingungen zu erfüllen
# richtige Lösung wird nicht angezeigt
k = 0.092
M = 25
C = 4.36
newtons_first = DE(name="newtons_first", input_min=-2., input_max=2.,
                   eq=lambda dT, T, x: dT - k * M + k * T,
                   order=1, ic_x=[0.], ic_y=[24.98722161],
                   solution=lambda x: M - (tf.exp(-C) * tf.exp(-k * x)))
equations.append(newtons_first)

# # ---------------------   second order   ----------------------------

# Simple Harmonic Motion (of spring) / Newton's Second Law
m = 1. / 16.
k1 = 4.
newtons_second_law = DE(name="newtons_second_law", input_min=-2., input_max=2.,
                        eq=lambda df_dx, df_dxx, f, x: m * df_dxx + k1 * f,
                        order=2, ic_x=[1.], ic_y=[-0.278346201920130888224993],
                        solution=lambda x: -2 * tf.sin(8 * x))
equations.append(newtons_second_law)

# x^2y′′+3xy′+4y=0
# Defintionslücke bei y(0)
c_1 = 5
c_2 = 3
# test case just to see what the NN comes up with for y(0) which is not defined
second_order_euler_test = DE(name="second_order_euler_test", input_min=-2., input_max=2.,
                             eq=lambda dy_dx, dy_dxx, y, x: tf.math.pow(x, 2) * dy_dxx + 3 * x * dy_dx + 4 * y,
                             order=2, ic_x=[1, 2.476632271], ic_y=[5, 0.4037741136],
                             solution=lambda x: c_1 * (1. / x) * tf.math.cos(tf.sqrt(3.) * tf.math.log(x)) + c_2 * (
                                     1. / x) * tf.math.sin(tf.sqrt(3.) * tf.math.log(x)))
equations.append(second_order_euler_test)

second_order_euler = DE(name="second_order_euler", input_min=2., input_max=6.,
                        eq=lambda dy_dx, dy_dxx, y, x: tf.math.pow(x, 2) * dy_dxx + 3 * x * dy_dx + 4 * y,
                        order=2, ic_x=[1, 2.476632271], ic_y=[5, 0.4037741136],
                        solution=lambda x: c_1 * (1. / x) * tf.math.cos(tf.sqrt(3.) * tf.math.log(x)) + c_2 * (
                                1. / x) * tf.math.sin(tf.sqrt(3.) * tf.math.log(x)))
equations.append(second_order_euler)

second_1 = DE(name="second_1", input_min=-2., input_max=2.,
              eq=lambda df_dx, df_dxx, f, x: 3 * ((x + 6.) ** 2.) * df_dxx + 25 * (x + 6.) * df_dx - 16 * f,
              order=2, ic_x=[-5., -4.], ic_y=[2, 1.591307302],
              solution=lambda x: tf.abs(x + 6.) ** (2. / 3.) + tf.abs(x + 6.) ** (-8.))
equations.append(second_1)

second_2 = DE(name="second_2", input_min=-2., input_max=2.,
              eq=lambda df_dx, df_dxx, x, t: df_dxx + x,
              order=2, ic_x=[0, 0.6366197724], ic_y=[1, 1],
              solution=lambda t: tf.cos(t) + tf.sin(t))
equations.append(second_2)

# # ------------------   third order ---------------------------


# third_order, y''' - 9y'' + 15y' + 25y = 0
third_order = DE(name="third_order", input_min=0., input_max=1.,
                 eq=lambda dy_dt, dy_dtt, dy_dttt, y, x: dy_dttt - 9 * dy_dtt + 15 * dy_dt + 25 * y,
                 order=3, ic_x=[0, 1, -1], ic_y=[3, 297.1941976, 2.718281828],
                 solution=lambda x: tf.math.exp(-x) + tf.math.exp(5 * x) + x * tf.math.exp(5 * x))
equations.append(third_order)

# third_order_2, y'''+y''-2y=e^x(14+34x+15x^2)
third_order_2 = DE(name="third_order_2", input_min=0., input_max=1.,
                   eq=lambda dy_dt, dy_dtt, dy_dttt, y, x: dy_dttt + dy_dtt - 2 * y - tf.math.exp(
                       14 + 34 * x + 15 * tf.math.pow(x, 2)),
                   order=3, ic_x=[0, 1.570796327, 1], ic_y=[2, 35.53210822, 8.529089278],
                   solution=lambda x: tf.math.exp(x) + tf.math.exp((-x)) * (
                           tf.math.cos(x) + tf.math.sin(x)) + tf.math.exp(x) * (
                                              tf.math.pow(x, 2) + tf.math.pow(x, 3)))
equations.append(third_order_2)

# third_order_3 y''' + y'' - 6y' + 4y = 0
third_order_3 = DE(name="third_order_3", input_min=0., input_max=1.,
                   eq=lambda dy_dt, dy_dtt, dy_dttt, y, x: dy_dttt + dy_dtt - 6 * dy_dt + 4 * y,
                   order=3, ic_x=[0, 1, 2], ic_y=[3, 6.199652613, 19.23832807],
                   solution=lambda x: tf.math.exp(x) + tf.math.exp((1.236067977) * x) + tf.math.exp((-3.236067977) * x))
equations.append(third_order_3)

# ###########################   nonlinear   #################################
# # ------------------------   first order   ---------------------------------

k2 = 0.07
L2 = 900
logistic_equation = DE(name="logistic_equation", input_min=-2., input_max=2.,
                       eq=lambda df_dx, f, x: df_dx - k2 * f * (1 - f / L2),
                       order=1, ic_x=[0], ic_y=[50],
                       solution=lambda x: 900 / (17 * tf.exp(-0.07 * x)))
equations.append(logistic_equation)

# nonlinear y' = x(y^3) where y(0)=2
nonlinear = DE(name="nonlinear", input_min=-2., input_max=2.,
               eq=lambda df_dx, y, x: df_dx - x * tf.math.pow(y, 3),
               order=1, ic_x=[0], ic_y=[2],
               solution=lambda x: tf.math.pow((1 / 4 - tf.math.pow(x, 2)), -0.5))
equations.append(nonlinear)

# # ------------------------   second order   ------------------------------------

# TODO add solution
# Painlevé II transcendent: w'' = 2w^3 + zw + α
alpha = 3.
painleve_2_transcendent = DE(name="painleve_2_transcendent", input_min=-2., input_max=2.,
               eq=lambda df_dx, df_dxx, y, x: 2 * (y**3) + x*y + alpha,
               order=2, ic_x=[4.7], ic_y=[0],
               solution=lambda x:None)
equations.append(painleve_2_transcendent)

second_order_nonlinear = DE(name="second_order_nonlinear", input_min=-2., input_max=2.,
               eq=lambda df_dx, df_dxx, f, x: -2. * x * (df_dx ** 2),
               order=2, ic_x=[0], ic_y=[2],
               solution=lambda x: 0.5 * (tf.math.log(tf.abs(x - 1.)) - tf.math.log(tf.abs(x + 1.))) + 2.)
equations.append(second_order_nonlinear)

# TODO: check ic
mu = 1.
van_der_pol = DE(name="van_der_pol", input_min=0., input_max=2,
                 eq=lambda dfdt, dfdtt, x, t: dfdtt - mu*(1-x**2)*dfdt + x,
                 order=2, ic_x=[0], ic_y=[2.],
                 solution=lambda x:None)
equations.append(van_der_pol)

# # ------------------------   third order   -----------------------------------

# third_order_nonlin, y′′′+(y′)^2−yy′′=0 
# x undefiniert, könnte zu Probemen führen. sind aber atsächlich  gleicvhverteielte x-Werte
third_order_nonlin = DE(name="third_order_nonlin", input_min=0., input_max=1.,
                        eq=lambda dy_dt, dy_dtt, dy_dttt, y, x: dy_dttt + tf.math.pow(dy_dt, 2) - y * dy_dtt,
                        order=3, ic_x=[0, 1, 2], ic_y=[1, 2.08616127, 6.524391382],
                        solution=lambda x: tf.math.exp(x) + tf.math.exp(-x) - 1)
equations.append(third_order_nonlin)

# third_order_v2, x^3(u''') - 3x^2(u'') + 7x(u') - 8u = f, while f = x^2/(1+ (ln|x|)^2), and f(0) = 0
A = 2
B = 2
C1 = 2
third_order_v2 = DE(name="third_order_v2", input_min=0., input_max=1.,
                    eq=lambda du_dx, du_dxx, du_dxxx, u, x: tf.math.pow(x, 3) * du_dxxx - 3 * (
                            x ** 2) * du_dxx + 7 * du_dx - 8 * u,
                    order=3, ic_x=[0, 1, 3], ic_y=[0, 2, 26.188934797822288],
                    solution=lambda x: (
                            A + B * tf.math.log(x) + C1 * (tf.math.log(x) ** 2) * (x ** 2) - ((x ** 2) / 2) * (
                            (1 - (tf.math.log(x)) ** 2) * tf.math.atan(tf.math.log(x)) + tf.math.log(
                        x) * tf.math.log((1 + (tf.math.log(x)) ** 2) - (tf.math.log(x))))))
equations.append(third_order_v2)

# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------

# set the hyperparameters
epochs = 10000
learning_rate = 0.01
loss_threshold = 0.00001

# lists for storing results
final_losses = []
final_errors = []
de_names = []

first_epoch_under_threshold = []
time_to_threshold = []
total_training_time = []

functions_dict = {}

input_size = 400
seed = 42

# create directory for the plots
try:
    os.makedirs(os.path.join(path_wd, "plots"))
except FileExistsError:
    # directory already exists
    pass

# create directory for the losses
try:
    os.makedirs(os.path.join(path_wd, "train_losses"))
except FileExistsError:
    # directory already exists
    pass

# create directory for the errors
try:
    os.makedirs(os.path.join(path_wd, "train_errors"))
except FileExistsError:
    # directory already exists
    pass


# save the hyperparameters
hyperparameters = []
hyperparameters.append(("epochs", epochs))
hyperparameters.append(("learning_rate", learning_rate))
hyperparameters.append(("loss_threshold", loss_threshold))
hyperparameters.append(("input_size", input_size))
hyperparameters.append(("seed", seed))
hyperparameters = np.array(hyperparameters)
np.savetxt(os.path.join(path_wd, "hyperparameters.txt"), hyperparameters, fmt="%s", delimiter=",")

# create RMSE function object for later use
rmse = tf.keras.metrics.RootMeanSquaredError()

# run the training and conversion for all DEs
for i, de in enumerate(equations):
    print("\n\nWorking on " + de.name + ", equation", i, "of", len(equations) - 1)
    # save the loss function in the dict
    functions_dict[de.name] = de.get_loss_function()

    # get the loss function and inputs from the DE object
    loss_function = de.get_loss_function()
    x = de.get_inputs(input_size)

    # Save dataset so SNN toolbox can find it.
    np.savez_compressed(os.path.join(path_wd, 'x_test'), x)
    np.savez_compressed(os.path.join(path_wd, 'y_test'), x)

    # initialize the model and optimizer
    ns = 10
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=ns, activation=tf.nn.silu,
                                                       kernel_initializer=tf.random_normal_initializer(seed=seed),
                                                       bias_initializer=tf.random_normal_initializer(seed=seed),
                                                       name="first",
                                                       input_shape=(1,)),
                                 tf.keras.layers.Dense(units=ns, activation=tf.nn.sigmoid,
                                                       kernel_initializer=tf.random_normal_initializer(seed=seed),
                                                       bias_initializer=tf.random_normal_initializer(seed=seed),
                                                       name="second"),
                                 tf.keras.layers.Dense(units=ns, activation=tf.nn.sigmoid,
                                                       kernel_initializer=tf.random_normal_initializer(seed=seed),
                                                       bias_initializer=tf.random_normal_initializer(seed=seed),
                                                       name="third"),
                                 tf.keras.layers.Dense(units=1, activation=tf.nn.selu,
                                                       kernel_initializer=tf.random_normal_initializer(seed=seed),
                                                       bias_initializer=tf.random_normal_initializer(seed=seed),
                                                       name="fourth")

                                 ])

    # store the model's activation functions for easier replication
    with open(os.path.join(path_wd, 'activations.txt'), 'w') as f:
        for layer in model.layers:
            try:
                print(layer.activation, file=f)
            except:  # some layers don't have any activation
                pass

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model.compile(optimizer, test_de.get_loss_function())
    print(model.summary())

    # Initialize lists for later visualization.
    train_losses = []
    train_errors = []

    under_threshold = False

    start_time = time.process_time()

    # We train for epochs.
    for epoch in range(epochs):
        # print the current loss every 100 epochs to check if training is working
        if epoch % 100 == 0:
            f = model(x)
            print(f'Epoch: {str(epoch)} starting with loss {loss_function(x, f)}')

        # run training and store the loss
        train_loss = training_step(model, x, loss_function, optimizer)
        train_losses.append(tf.squeeze(train_loss))

        # calculate error and store it
        # only possible if solution is not None
        try:
            approx = tf.squeeze(model(x))
            solution = tf.squeeze(de.analytical_solution(tf.squeeze(x)))
            error = rmse(approx, solution).numpy()
            train_errors.append(error)
        except ValueError:
            pass

        # check if loss is under threshold
        if tf.squeeze(train_loss) < loss_threshold and not under_threshold:
            time_to_threshold.append(time.process_time() - start_time)
            first_epoch_under_threshold.append(epoch)
            under_threshold = True

    # store total runtime and None for time to threshold if it was not reached (to have equal sized lists for all DEs)
    total_training_time.append(time.process_time() - start_time)
    if not under_threshold:
        time_to_threshold.append(None)
        first_epoch_under_threshold.append(None)

    # save final loss + error
    if train_errors:
        final_errors.append(train_errors[-1])
    else:
        final_errors.append(None)
    final_losses.append(train_losses[-1])
    de_names.append(de.name)

    # save the model
    model_name = de.name
    model.save(os.path.join(path_wd, model_name))

    # save train loss and train error to file for later evaulation
    np.savetxt(os.path.join(path_wd, "train_losses", model_name+"-train_losses.txt"), np.array(train_losses), fmt="%s", delimiter=",")
    np.savetxt(os.path.join(path_wd, "train_errors", model_name+"-train_errors.txt"), np.array(train_errors), fmt="%s", delimiter=",")

    # plot result
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.figure()
    plt.plot(train_losses)
    # if train_errors:
    #     plt.plot(train_errors, label="RMSE")
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.title(de.name)
    plt.legend()

    figname = de.name + "__loss.png"
    plt.savefig(os.path.join(path_wd, "plots", figname))
    plt.show(block=False)
    plt.pause(0.001)

    # plot result
    if train_errors:
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.figure()
        plt.plot(train_errors)
        plt.xlabel("Training steps")
        plt.ylabel("RMSE")
        plt.title(de.name)
        plt.legend()

        figname = de.name + "__error.png"
        plt.savefig(os.path.join(path_wd, "plots", figname))
        plt.show(block=False)
        plt.pause(0.001)

    # plot the model's approximation and the actual solution
    approx = model(x)
    plt.plot(tf.squeeze(x), tf.squeeze(approx), label="model's solution")
    try:
        plt.plot(tf.squeeze(x), tf.squeeze(de.analytical_solution(tf.squeeze(x))), label="true solution",
                 linestyle="dashed")
    except ValueError:
        pass
    plt.legend()
    plt.title(de.name)

    figname = de.name + "__solution.png"
    plt.savefig(os.path.join(path_wd, "plots", figname))
    plt.show(block=False)
    plt.pause(0.001)

    # print and store the dictionary with the loss functions
    print("dictionary:", functions_dict)
    dict_file_name = 'serialized_custom_loss_functions.txt'
    f = open(os.path.join(path_wd, dict_file_name), 'wb')  # opened the file in write and binary mode
    pickle.dump(functions_dict, f)  # dumping the content in the variable 'content' into the file
    f.close()  # closing the file



# create an array with all final losses + errors and de names
final_losses = np.array(final_losses)
final_errors = np.array(final_errors)
de_names = np.array(de_names)
time_to_threshold = np.array(time_to_threshold)
total_training_time = np.array(total_training_time)
first_epoch_under_threshold = np.array(first_epoch_under_threshold)

metrics = np.array(
    [de_names, final_losses, final_errors, first_epoch_under_threshold, time_to_threshold, total_training_time]).T

# save metrics as file
np.savetxt(os.path.join(path_wd, "metrics.txt"), metrics, fmt="%s", delimiter=",")

# show all plots in the end when running as script
plt.show()
