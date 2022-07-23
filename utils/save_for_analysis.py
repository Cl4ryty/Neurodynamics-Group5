import os
import numpy as np
import pandas as pd

# path to the data
path = os.path.join(os.getcwd(), "..", "data", "1657861020.4330547")

# define categories:
linear = ["test_de", "gompertz", "kirchhoff", "newtons_first", "newtons_second_law", "second_order_euler_test",
          "second_order_euler",
          "second_1", "second_2", "new_2nd_linear_1", "new_2nd_linear_2", "new_2nd_linear_3", "third_order",
          "third_order_2", "third_order_3"]
nonlinear = ["logistic_equation", "nonlinear", "painleve_2_transcendent", "second_order_nonlinear", "van_der_pol",
             "new_2nd_nonlinear_1", "new_2nd_nonlinear_2", "third_order_nonlin", "third_order_v2"]
first_order = ["test_de", "gompertz", "kirchhoff", "newtons_first", "logistic_equation", "nonlinear"]
second_order = ["newtons_second_law", "second_order_euler_test", "second_order_euler", "second_1",
                "second_2", "new_2nd_linear_1", "new_2nd_linear_2", "new_2nd_linear_3", "painleve_2_transcendent",
                "second_order_nonlinear", "van_der_pol", "new_2nd_nonlinear_1", "new_2nd_nonlinear_2"]
third_order = ["third_order", "third_order_2", "third_order_3", "third_order_nonlin", "third_order_v2"]

# functions to determine linearity and order of DE
is_linear = lambda x: 1 if x in linear else 0


def get_order(x):
    if x in first_order:
        return 1
    if x in second_order:
        return 2
    if x in third_order:
        return 3
    return -1

# load time metrics
time_metrics_path = os.path.join(path, "run_metrics.npy")
time_metrics = np.load(time_metrics_path, allow_pickle=True)
de_names = time_metrics[0,:,0]

# add category columns
time_metrics = np.insert(time_metrics, time_metrics.shape[2], values=np.array([is_linear(x) for x in de_names]), axis=2)
time_metrics = np.insert(time_metrics, time_metrics.shape[2], values=np.array([get_order(x) for x in de_names]), axis=2)

# turn 3d numpy array into pandas dataframe
m,n,r = time_metrics.shape
out_arr = np.column_stack((np.repeat(np.arange(m),n),time_metrics.reshape(m*n,-1)))
out_df = pd.DataFrame(out_arr)

# add header for columns:
# de_names, final_losses, final_errors, first_epoch_under_threshold, time_to_threshold, total_training_time, is_linear, order
time_metrics = pd.DataFrame(out_arr, columns=["run_nb", "de_names", "final_losses", "final_errors", "first_epoch_under_threshold", "time_to_threshold", "total_training_time", "is_linear", "order"])
# change column data types
time_metrics[["final_losses", "final_errors", "first_epoch_under_threshold", "time_to_threshold", "total_training_time", "order", "is_linear"]] = time_metrics[["final_losses", "final_errors", "first_epoch_under_threshold", "time_to_threshold", "total_training_time", "order", "is_linear"]].apply(pd.to_numeric, errors='coerce')
time_metrics["de_names"] = time_metrics["de_names"].astype(str)
time_metrics["is_linear"] = time_metrics["is_linear"].astype(bool)

print(time_metrics)
# save as csv for analysis / further plotting
time_metrics.to_csv(os.path.join("..", "Analysis", "time_metrics.csv"))

