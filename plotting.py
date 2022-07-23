import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pretty_plots import pretty_plot_settings

# make plots look better
pretty_plot_settings()

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

# All metrics except time:
# import data from one run - metrics.txt, (train_errors folder, train_losses folder)

path = os.path.join(os.getcwd(), "data", "1657861020.4330547")
metrics_path = os.path.join(path, "0", "metrics.txt")
metrics = np.loadtxt(metrics_path, delimiter=",", dtype=str)
de_names = metrics[:, 0]

is_linear = lambda x: 1 if x in linear else 0


def get_order(x):
    if x in first_order:
        return 1
    if x in second_order:
        return 2
    if x in third_order:
        return 3
    return -1


# add category columns
metrics = np.insert(metrics, metrics.shape[1], values=[is_linear(x) for x in de_names], axis=1)
metrics = np.insert(metrics, metrics.shape[1], values=[get_order(x) for x in de_names], axis=1)

# columns:
# de_names, final_losses, final_errors, first_epoch_under_threshold, time_to_threshold, total_training_time, is_linear, order
metrics = pd.DataFrame(metrics, columns=["de_names", "final_losses", "final_errors", "first_epoch_under_threshold", "time_to_threshold", "total_training_time", "is_linear", "order"])
metrics[["final_losses", "final_errors", "first_epoch_under_threshold", "time_to_threshold", "total_training_time", "order", "is_linear"]] = metrics[["final_losses", "final_errors", "first_epoch_under_threshold", "time_to_threshold", "total_training_time", "order", "is_linear"]].apply(pd.to_numeric, errors='coerce')
metrics["de_names"] = metrics["de_names"].astype(str)
metrics["is_linear"] = metrics["is_linear"].astype(bool)


print(metrics)
print(metrics.dtypes)

print(metrics.describe())

# exclude DEs with very high loss
df = metrics[metrics['final_losses']<10]
print(df)


# plots for final loss
plt.figure(figsize=(15, 8))
plt.bar(df["de_names"].tolist(), df["final_losses"].tolist())
plt.ylabel("Final losses")
plt.xlabel("DEs")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)


# -------------------------------- grouped by order --------------------------------------------------------------------


# plots for final loss
# mean
plt.figure(figsize=(15, 8))
df.groupby("order").mean()["final_losses"].plot.bar()
plt.ylabel("Mean final losses")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# median
plt.figure(figsize=(15, 8))
df.groupby("order").median()["final_losses"].plot.bar()
plt.ylabel("Median final losses")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# min
plt.figure(figsize=(15, 8))
df.groupby("order").min()["final_losses"].plot.bar()
plt.ylabel("Min final losses")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# max
plt.figure(figsize=(15, 8))
df.groupby("order").max()["final_losses"].plot.bar()
plt.ylabel("Max final losses")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)


# ----------------------------- linear vs nonlinear --------------------------------------------------------------------

# mean
plt.figure(figsize=(15, 8))
df.groupby("is_linear").mean()["final_losses"].plot.bar()
plt.ylabel("Mean final losses")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# median
plt.figure(figsize=(15, 8))
df.groupby("is_linear").median()["final_losses"].plot.bar()
plt.ylabel("Median final losses")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# min
plt.figure(figsize=(15, 8))
df.groupby("is_linear").min()["final_losses"].plot.bar()
plt.ylabel("Min final losses")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# max
plt.figure(figsize=(15, 8))
df.groupby("is_linear").max()["final_losses"].plot.bar()
plt.ylabel("Max final losses")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)


# ----------------------------- grouped by linearity and order --------------------------------------------------------------------

# mean
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).mean()["final_losses"].plot.bar()
plt.ylabel("Mean final losses")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# median
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).median()["final_losses"].plot.bar()
plt.ylabel("Median final losses")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# min
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).min()["final_losses"].plot.bar()
plt.ylabel("Min final losses")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# max
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).max()["final_losses"].plot.bar()
plt.ylabel("Max final losses")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)





print(df.groupby(["is_linear", "order"]).median())
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).median()["final_losses"].plot.bar()
plt.ylabel("Final losses")
plt.xlabel("DEs")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)






# time metrics
time_metrics_path = os.path.join(path, "run_metrics.npy")
time_metrics = np.load(time_metrics_path, allow_pickle=True)
de_names = time_metrics[0,:,0]
print(de_names)
print("time_metrics", time_metrics)
print(time_metrics.shape)
# add category columns
time_metrics = np.insert(time_metrics, time_metrics.shape[2], values=np.array([is_linear(x) for x in de_names]), axis=2)
time_metrics = np.insert(time_metrics, time_metrics.shape[2], values=np.array([get_order(x) for x in de_names]), axis=2)

print("time_metrics", time_metrics)


# columns:
# de_names, final_losses, final_errors, first_epoch_under_threshold, time_to_threshold, total_training_time, is_linear, order

m,n,r = time_metrics.shape
out_arr = np.column_stack((np.repeat(np.arange(m),n),time_metrics.reshape(m*n,-1)))
out_df = pd.DataFrame(out_arr)

time_metrics = pd.DataFrame(out_arr, columns=["run_nb", "de_names", "final_losses", "final_errors", "first_epoch_under_threshold", "time_to_threshold", "total_training_time", "is_linear", "order"])
time_metrics[["final_losses", "final_errors", "first_epoch_under_threshold", "time_to_threshold", "total_training_time", "order", "is_linear"]] = time_metrics[["final_losses", "final_errors", "first_epoch_under_threshold", "time_to_threshold", "total_training_time", "order", "is_linear"]].apply(pd.to_numeric, errors='coerce')
time_metrics["de_names"] = time_metrics["de_names"].astype(str)
time_metrics["is_linear"] = time_metrics["is_linear"].astype(bool)

print(time_metrics)
df = time_metrics


# #################################### total training time #############################################################

plt.figure(figsize=(15, 8))
df.groupby(["is_linear"]).median()["total_training_time"].plot.bar()
plt.ylabel("Median total training time")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)


# exclude DEs with very high loss
df = metrics[metrics['final_losses']<10]


# plots for final loss
# plt.figure(figsize=(15, 8))
# plt.boxplot(labels=de_names, x=df["total_training_time"].tolist())
# plt.ylabel("Total training times")
# plt.xlabel("DEs")
# plt.xticks(rotation=30, ha='right')
# # plt.yscale('log')
# # figname = "Total_training_times.png"
# # plt.savefig(os.path.join(top_path, "plots", figname))
# plt.show(block=False)
# plt.pause(0.001)


# -------------------------------- grouped by order --------------------------------------------------------------------


# plots for final loss
# mean
plt.figure(figsize=(15, 8))
df.groupby("order").mean()["total_training_time"].plot.bar()
plt.ylabel("Mean total training time")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# median
plt.figure(figsize=(15, 8))
df.groupby("order").median()["total_training_time"].plot.bar()
plt.ylabel("Median total training time")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# min
plt.figure(figsize=(15, 8))
df.groupby("order").min()["total_training_time"].plot.bar()
plt.ylabel("Min total training time")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# max
plt.figure(figsize=(15, 8))
df.groupby("order").max()["total_training_time"].plot.bar()
plt.ylabel("Max total training time")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)



# ----------------------------- linear vs nonlinear --------------------------------------------------------------------

# mean
plt.figure(figsize=(15, 8))
df.groupby("is_linear").mean()["total_training_time"].plot.bar()
plt.ylabel("Mean total training time")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# median
plt.figure(figsize=(15, 8))
df.groupby("is_linear").median()["total_training_time"].plot.bar()
plt.ylabel("Median total training time")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# min
plt.figure(figsize=(15, 8))
df.groupby("is_linear").min()["total_training_time"].plot.bar()
plt.ylabel("Min total training time")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# max
plt.figure(figsize=(15, 8))
df.groupby("is_linear").max()["total_training_time"].plot.bar()
plt.ylabel("Max total training time")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)


# ----------------------------- grouped by linearity and order --------------------------------------------------------------------

# mean
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).mean()["total_training_time"].plot.bar()
plt.ylabel("Mean total training time")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# median
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).median()["total_training_time"].plot.bar()
plt.ylabel("Median total training time")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# min
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).min()["total_training_time"].plot.bar()
plt.ylabel("Min total training time")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# max
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).max()["total_training_time"].plot.bar()
plt.ylabel("Max total training time")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)



# ################################### time to threshold ################################################################

plt.figure(figsize=(15, 8))
df.groupby(["is_linear"]).median()["time_to_threshold"].plot.bar()
plt.ylabel("Median time to threshold")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)


# exclude DEs with very high loss
df = metrics[metrics['final_losses']<10]


# plots for final loss
# plt.figure(figsize=(15, 8))
# plt.boxplot(labels=df["de_names"].tolist(), x=df["time_to_threshold"].tolist())
# plt.ylabel("Total time to threshold")
# plt.xlabel("DEs")
# plt.xticks(rotation=30, ha='right')
# # plt.yscale('log')
# # figname = "Total_training_times.png"
# # plt.savefig(os.path.join(top_path, "plots", figname))
# plt.show(block=False)
# plt.pause(0.001)


# -------------------------------- grouped by order --------------------------------------------------------------------


# plots for final loss
# mean
plt.figure(figsize=(15, 8))
df.groupby("order").mean()["time_to_threshold"].plot.bar()
plt.ylabel("Mean time to threshold")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# median
plt.figure(figsize=(15, 8))
df.groupby("order").median()["time_to_threshold"].plot.bar()
plt.ylabel("Median time to threshold")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# min
plt.figure(figsize=(15, 8))
df.groupby("order").min()["time_to_threshold"].plot.bar()
plt.ylabel("Min time to threshold")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# max
plt.figure(figsize=(15, 8))
df.groupby("order").max()["time_to_threshold"].plot.bar()
plt.ylabel("Max time to threshold")
plt.xlabel("DE Order")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)



# ----------------------------- linear vs nonlinear --------------------------------------------------------------------

# mean
plt.figure(figsize=(15, 8))
df.groupby("is_linear").mean()["time_to_threshold"].plot.bar()
plt.ylabel("Mean time to threshold")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# median
plt.figure(figsize=(15, 8))
df.groupby("is_linear").median()["time_to_threshold"].plot.bar()
plt.ylabel("Median time to threshold")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# min
plt.figure(figsize=(15, 8))
df.groupby("is_linear").min()["time_to_threshold"].plot.bar()
plt.ylabel("Min time to threshold")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# max
plt.figure(figsize=(15, 8))
df.groupby("is_linear").max()["time_to_threshold"].plot.bar()
plt.ylabel("Max time to threshold")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)


# ----------------------------- grouped by linearity and order --------------------------------------------------------------------

# mean
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).mean()["time_to_threshold"].plot.bar()
plt.ylabel("Mean time to threshold")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# median
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).median()["time_to_threshold"].plot.bar()
plt.ylabel("Median time to threshold")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# min
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).min()["time_to_threshold"].plot.bar()
plt.ylabel("Min time to threshold")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)

# plots for final loss
# max
plt.figure(figsize=(15, 8))
df.groupby(["is_linear", "order"]).max()["time_to_threshold"].plot.bar()
plt.ylabel("Max time to threshold")
plt.xlabel("Is linear")
plt.xticks(rotation=30, ha='right')
# plt.yscale('log')
# figname = "Total_training_times.png"
# plt.savefig(os.path.join(top_path, "plots", figname))
plt.show(block=False)
plt.pause(0.001)




