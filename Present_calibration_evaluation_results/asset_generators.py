import itertools
import json
import math
from multiprocessing.dummy import active_children
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import statistics
import sys
import warnings
warnings.filterwarnings("ignore")

res_addition_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_ff.json")
res_addition_ff_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_ff_ln.json")
res_addition_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_tf.json")
res_addition_tf_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_tf_ln.json")
res_addition_tt = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_tt.json")
res_addition_tt_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_tt_ln.json")
res_noisy_addition_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_noisy_addition_ff.json")
res_noisy_addition_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_noisy_addition_tf.json")
res_noisy_addition_tt = os.path.join("calibration_evaluator_results", "calibration_evaluation_noisy_addition_tt.json")
res_hwf_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_hwf_ff.json")
res_hwf_ff_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_hwf_ff_ln.json")
res_hwf_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_hwf_tf.json")
res_hwf_tf_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_hwf_tf_ln.json")
res_hwf_tt = os.path.join("calibration_evaluator_results", "calibration_evaluation_hwf_tt.json")
res_hwf_tt_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_hwf_tt_ln.json")
res_coins_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_coins_ff.json")
res_coins_ff_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_coins_ff_ln.json")
res_coins_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_coins_tf.json")
res_coins_tf_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_coins_tf_ln.json")
res_coins_tt = os.path.join("calibration_evaluator_results", "calibration_evaluation_coins_tt.json")
res_coins_tt_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_coins_tt_ln.json")
res_poker_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_poker_ff.json")
res_poker_ff_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_poker_ff_ln.json")
res_poker_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_poker_tf.json")
res_poker_tf_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_poker_tf_ln.json")
res_poker_tt = os.path.join("calibration_evaluator_results", "calibration_evaluation_poker_tt.json")
res_poker_tt_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_poker_tt_ln.json")
res_forth_add_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_add_ff.json")
res_forth_add_ff_ln =os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_add_ff_ln.json")
res_forth_add_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_add_tf.json")
res_forth_add_tf_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_add_tf_ln.json")
res_forth_add_tt = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_add_tt.json")
res_forth_add_tt_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_add_tt_ln.json")
res_forth_sort_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_sort_ff.json")
res_forth_sort_ff_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_sort_ff_ln.json")
res_forth_sort_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_sort_tf.json")
res_forth_sort_tf_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_sort_tf_ln.json")
res_forth_sort_tt = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_sort_tt.json")
res_forth_sort_tt_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_sort_tt_ln.json")
res_forth_wap_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_wap_ff.json")
res_forth_wap_ff_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_wap_ff_ln.json")
res_forth_wap_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_wap_tf.json")
res_forth_wap_tf_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_forth_wap_tf_ln.json")
res_clutrr_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_clutrr_ff.json")
res_clutrr_ff_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_clutrr_ff_ln.json")
res_clutrr_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_clutrr_tf.json")
res_clutrr_tf_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_clutrr_tf_ln.json")

result_objects = [
  ("MNIST addition", False, False, False, res_addition_ff),
  ("MNIST addition", False, False, True, res_addition_ff_ln),
  ("MNIST addition", True, False, False, res_addition_tf),
  ("MNIST addition", True, False, True, res_addition_tf_ln),
  ("MNIST addition", True, True, False, res_addition_tt),
  ("MNIST addition", True, True, True, res_addition_tt_ln),
  ("Noisy MNIST addition", False, False, False, res_noisy_addition_ff),
  ("Noisy MNIST addition", True, False, False, res_noisy_addition_tf),
  ("Noisy MNIST addition", True, True, False, res_noisy_addition_tt),
  ("HWF", False, False, False, res_hwf_ff),
  ("HWF", False, False, True, res_hwf_ff_ln),
  ("HWF", True, False, False, res_hwf_tf),
  ("HWF", True, False, True, res_hwf_tf_ln),
  ("HWF", True, True, False, res_hwf_tt),
  ("HWF", True, True, True, res_hwf_tt_ln),
  ("Coins", False, False, False, res_coins_ff),
  ("Coins", False, False, True, res_coins_ff_ln),
  ("Coins", True, False, False, res_coins_tf),
  ("Coins", True, False, True, res_coins_tf_ln),
  ("Coins", True, True, False, res_coins_tt),
  ("Coins", True, True, True, res_coins_tt_ln),
  ("Poker", False, False, False, res_poker_ff),
  ("Poker", False, False, True, res_poker_ff_ln),
  ("Poker", True, False, False, res_poker_tf),
  ("Poker", True, False, True, res_poker_tf_ln),
  ("Poker", True, True, False, res_poker_tt),
  ("Poker", True, True, True, res_poker_tt_ln),
  ("Forth/Add", False, False, False, res_forth_add_ff),
  ("Forth/Add", False, False, True, res_forth_add_ff_ln),
  ("Forth/Add", True, False, False, res_forth_add_tf),
  ("Forth/Add", True, False, True, res_forth_add_tf_ln),
  ("Forth/Add", True, True, False, res_forth_add_tt),
  ("Forth/Add", True, True, True, res_forth_add_tt_ln),
  ("Forth/Sort", False, False, False, res_forth_sort_ff),
  ("Forth/Sort", False, False, True, res_forth_sort_ff_ln),
  ("Forth/Sort", True, False, False, res_forth_sort_tf),
  ("Forth/Sort", True, False, True, res_forth_sort_tf_ln),
  ("Forth/Sort", True, True, False, res_forth_sort_tt),
  ("Forth/Sort", True, True, True, res_forth_sort_tt_ln),
  ("Forth/WAP", False, False, False, res_forth_wap_ff),
  ("Forth/WAP", False, False, True, res_forth_wap_ff_ln),
  ("Forth/WAP", True, False, False, res_forth_wap_tf),
  ("Forth/WAP", True, False, True, res_forth_wap_tf_ln),
  ("CLUTRR", False, False, False, res_clutrr_ff),
  ("CLUTRR", False, False, True, res_clutrr_ff_ln),
  ("CLUTRR", True, False, False, res_clutrr_tf),
  ("CLUTRR", True, False, True, res_clutrr_tf_ln),
]

def safe_stdev(listlike):
  return statistics.stdev(listlike) if len(listlike) > 1 else 0

def setup_rc(align_ticks = False):
  plt.rc('xtick', labelsize = "medium") 
  plt.rc('ytick', labelsize = "medium")
  plt.rc('axes', labelsize = "medium")
  plt.rc("legend", fontsize = "medium")
  plt.rc("font", size = 9.0)
  if align_ticks:
    plt.rc("xtick", alignment = "left")

def get_experiment_name(calibrated, calibrated_after_each_train_iteration, label_noise_added):
  experiment_name = ""
  if calibrated:
    if calibrated_after_each_train_iteration:
      experiment_name += "C.A.E.I."
    else:
      experiment_name += "C."
  else:
    experiment_name += "U."
  if label_noise_added:
    experiment_name += ", A. L. N."

  return experiment_name

def generate_accuracies_table():
  accuracies_table = []
  for example_name, calibrated, calibrated_after_each_train_iteration, label_noise_added, result_file in result_objects:
    with open(result_file, "r") as result_file_fd:
      result_object = json.load(result_file_fd)
      accuracy = np.NaN
      if "accuracy" in result_object:
        accuracy = result_object["accuracy"]
      elif "accuracies" in result_object:
        accuracy = statistics.mean(result_object["accuracies"])

      accuracies_row = (
        example_name,
        get_experiment_name(calibrated, calibrated_after_each_train_iteration, label_noise_added),
        accuracy
      )
      accuracies_table.append(accuracies_row)

  return pd.DataFrame(
    accuracies_table,
    columns = ["Example", "Experiment", "Accuracy"]
  )

def plot_accuracies(accuracies_table, figure_filename):
  sns.set_context('poster', font_scale = 1.4)
  sns.set_theme(color_codes = True)
  setup_rc(align_ticks = True)
  plt.gcf().set_figwidth(6)
  g = sns.barplot(x = "Example", y = "Accuracy", hue = "Experiment", data = accuracies_table)
  ticklabel_texts = accuracies_table["Example"].unique().tolist()
  g.set_xticklabels(ticklabel_texts, visible = True, rotation = -30.0)
  g.tick_params(bottom = True, labelbottom = True, top = False, labeltop = False)

  plt.tight_layout()
  plt.savefig(figure_filename)

def generate_ECE_difference_table():
  ECEs_table = []
  for example_name, calibrated, calibrated_after_each_train_iteration, label_noise_added, result_file in result_objects:
    if not calibrated:
      continue
    with open(result_file, "r") as result_file_fd:
      result_object = json.load(result_file_fd)
      if calibrated_after_each_train_iteration:
        calibration_collector = result_object["networks_evolution_collectors"]["calibration_collector"]
        before_calibration_ece_histories = calibration_collector["before_calibration_ece_history"]
        after_calibration_ece_histories = calibration_collector["after_calibration_ece_history"]
        before_last_calibration_mean_ece = statistics.mean(
          map(lambda history: history[-1], before_calibration_ece_histories.values())
        )
        before_last_calibration_stdev_ece = safe_stdev(
          list(map(lambda history: history[-1], before_calibration_ece_histories.values()))
        )
        after_last_calibration_mean_ece = statistics.mean(
          map(lambda history: history[-1], after_calibration_ece_histories.values())
        )
        after_last_calibration_stdev_ece = safe_stdev(
          list(map(lambda history: history[-1], after_calibration_ece_histories.values()))
        )
      else:
        ECEs_final_calibration = result_object["ECEs_final_calibration"]
        before_last_calibration_mean_ece = statistics.mean(
          map(lambda v: v["before"], ECEs_final_calibration.values())
        )
        before_last_calibration_stdev_ece = safe_stdev(
          list(map(lambda v: v["before"], ECEs_final_calibration.values()))
        )
        after_last_calibration_mean_ece = statistics.mean(
          map(lambda v: v["after"], ECEs_final_calibration.values())
        )
        after_last_calibration_stdev_ece = safe_stdev(
          list(map(lambda v: v["after"], ECEs_final_calibration.values()))
        )

      ECEs_row = (
        example_name,
        get_experiment_name(calibrated, calibrated_after_each_train_iteration, label_noise_added),
        before_last_calibration_mean_ece,
        before_last_calibration_stdev_ece,
        after_last_calibration_mean_ece,
        after_last_calibration_stdev_ece
      )
      ECEs_table.append(ECEs_row)

  return pd.DataFrame(
    ECEs_table,
    columns = ["Example", "Experiment", "ECE before calibration μ", "ECE before calibration σ", "ECE after calibration μ", "ECE after calibration σ"]
  )

def plot_ECE_difference(ECE_differences_table, figure_filename):
  data = pd.melt(ECE_differences_table, ["Example", "Experiment"])

  sns.set_context('poster', font_scale = 1.4)
  sns.set_theme(color_codes = True)
  setup_rc()
  plt.gcf().set_figwidth(6)
  g = sns.catplot(x = "Experiment", y = "value", hue = "variable", col = "Example", col_wrap = 4, data = data, kind = 'bar', facet_kws = {"sharex": False})

  g._legend.set_title('')
  ticklabel_texts = [
    ticklabel.get_text()
    for ticklabel in g.axes[-1].get_xticklabels()
  ]
  ticks = g.axes[-1].get_xticks().astype('float16').tolist()

  for i in range(g.axes.shape[0]):
    g.axes[i].set_title(
       g.axes[i].get_title()[10:]
    )
    g.axes[i].set_ylabel("")
    g.axes[i].set_xticks(ticks)
    g.axes[i].set_xticklabels(ticklabel_texts, visible = True)
    g.axes[i].tick_params(bottom = True, labelbottom = True, top = False, labeltop = False)

  plt.tight_layout()
  plt.savefig(figure_filename)

def generate_loss_ECE_evolution_table():
  loss_ECE_evolution = {
    'Example': [],
    "Experiment": [],
    "Iteration": [],
    "Cross-entropy loss": [],
    "ECE μ": [],
  }
  for example_name, calibrated, calibrated_after_each_train_iteration, label_noise_added, result_file in result_objects:
    if calibrated:
      with open(result_file, "r") as result_file_fd:
        result_object = json.load(result_file_fd)
        loss_history = result_object["loss_history"]
        calibration_collector = result_object["networks_evolution_collectors"]["calibration_collector"]
        ece_histories = calibration_collector["ece_history"]        
        average_ece_history = list(map(
          statistics.mean,
          zip(*ece_histories.values())
        ))

        for iteration, loss, average_ece in zip(
          range(1, len(loss_history) + 1),
          loss_history,
          average_ece_history
        ):
          loss_ECE_evolution["Example"].append(example_name)
          loss_ECE_evolution["Experiment"].append(get_experiment_name(calibrated, calibrated_after_each_train_iteration, label_noise_added))
          loss_ECE_evolution["Iteration"].append(iteration)
          loss_ECE_evolution["Cross-entropy loss"].append(loss)
          loss_ECE_evolution["ECE μ"].append(average_ece)
  
  return pd.DataFrame(loss_ECE_evolution)

def plot_loss_ECE_evolution(loss_ECE_evolution_table, figure_filename):
  loss_ECE_evolution_data = pd.melt(loss_ECE_evolution_table, ["Example", "Experiment", "Iteration"])
  cols = loss_ECE_evolution_data["Experiment"].unique().tolist()
  rows = loss_ECE_evolution_data["Example"].unique().tolist()

  sns.set_context('poster', font_scale = 1.4)
  sns.set_theme(color_codes = True)
  plt.gcf().set_figwidth(6)
  g = sns.lmplot(x = 'Iteration', y = 'value', hue = 'variable', col = 'Experiment', row = "Example", data = loss_ECE_evolution_data, order = 5, ci = False, legend = True, facet_kws = {"sharex": False, "sharey": False})

  for i, j in itertools.product(range(g.axes.shape[0]), range(g.axes.shape[1])):
    if len(g.axes[i, j].lines) > 0:
      data_1 = g.axes[i, j].lines[0].get_data()
      data_2 = g.axes[i, j].lines[1].get_data()
      g.axes[i, j].set_yscale('symlog', linthresh = min(max(data_1[1]), max(data_2[1])))
    g.axes[i, j].set_title("")
  for i in range(g.axes.shape[0]):
    g.axes[i, 0].set_ylabel("")

  pad = 5
  for ax, col in zip(g.axes[0], cols):
    ax.annotate(
      col,
      xy = (0.5, 1),
      xytext = (0, pad),
      xycoords = 'axes fraction',
      textcoords = 'offset points',
      size = 'large',
      ha = 'center',
      va = 'baseline'
    )
  for ax, row in zip(g.axes[:, 0], rows):
    ax.annotate(
      row,
      xy = (0, 0.5),
      xytext = (-ax.yaxis.labelpad - pad, 0),
      xycoords = ax.yaxis.label,
      textcoords = 'offset points',
      size = 'large',
      ha = 'right',
      va = 'center'
    )
  g._legend.set_title('')

  nas = [
    g.axes[1, 1],
    g.axes[1, 3],
    g.axes[7, 2],
    g.axes[7, 3],
    g.axes[8, 2],
    g.axes[8, 3]
  ]
  for na in nas:
    na.clear()

  na_xlim = g.axes[1, 1].get_xlim()
  na_ylim = g.axes[1, 1].get_ylim()
  na_center_x = na_xlim[0] + ((na_xlim[1] - na_xlim[0]) / 2)
  na_center_y = na_ylim[0] + ((na_ylim[1] - na_ylim[0]) / 2)
  for na in nas:
    na.text(na_center_x, na_center_y, "N.A.", ha = "center", va = "center")

  xlabel = g.axes[8, 1].get_xlabel()
  g.axes[8, 2].set_xlabel(xlabel)
  g.axes[8, 3].set_xlabel(xlabel)

  plt.tight_layout()
  plt.savefig(figure_filename)

def analyze_accuracies_table(accuracies_table):
  examples = accuracies_table["Example"].unique().tolist()
  experiments = accuracies_table["Experiment"].unique().tolist()
  experiments_without_label_noise = list(filter(
    lambda x: not x.endswith("A. L. N."), 
    experiments
  ))
  experiments_with_label_noise = list(filter(
    lambda x: x.endswith("A. L. N."), 
    experiments
  ))
  def get_iteration_product(experiments):
    return filter(
      lambda p: \
        (not (p[0].startswith("Noisy") and p[1].endswith("A.L.N."))) and \
        (not (p[0] == "Forth/WAP" and p[1].startswith("C.A.E.I."))) and \
        (not (p[0] == "CLUTRR" and p[1].startswith("C.A.E.I."))),
      itertools.product(examples, experiments),
    )

  print("- Analyzing accuracies table -")
  print("-- Maximum accuracy experiment counts:")
  row_with_max_accuracy = lambda group: group.iloc[group["Accuracy"].argmax()]
  max_accuracies = accuracies_table.groupby(["Example"]).apply(row_with_max_accuracy)
  experiment_value_counts = max_accuracies["Experiment"].value_counts()
  for k, v in experiment_value_counts.iteritems():
    print(f"{k} - {v}")

  print("-- Accuracy differences without label noise:")
  def acc_diffs(group):
    current_example = group["Example"].iloc[0]
    cols = [current_example]
    uncalibrated_acc = group[group["Experiment"] == "U."]["Accuracy"].iloc[0]
    cols.append(uncalibrated_acc)
    cols.append(group[group["Experiment"] == "C."]["Accuracy"].iloc[0] - uncalibrated_acc)
    if current_example != "Forth/WAP" and \
       current_example != "CLUTRR":
      cols.append(group[group["Experiment"] == "C.A.E.I."]["Accuracy"].iloc[0] - uncalibrated_acc)
    else:
      cols.append(float('nan'))
    return pd.Series(cols, ["Experiment", "U.", "C.", "C.A.E.I."])
  accuracies_table_ = accuracies_table[~accuracies_table["Experiment"].str.endswith("A. L. N.")]
  accuracies_table__ = accuracies_table_.groupby(["Example"]).apply(acc_diffs)
  print(accuracies_table__)
  print(accuracies_table__.describe())

  print("-- Accuracy differences when label noise is added:")
  def acc_diffs_when_ln_added(group):
    current_example = group["Example"].iloc[0]
    cols = [current_example]
    uncalibrated_acc = group[group["Experiment"] == "U., A. L. N."]["Accuracy"].iloc[0]
    cols.append(uncalibrated_acc)
    cols.append(group[group["Experiment"] == "C., A. L. N."]["Accuracy"].iloc[0] - uncalibrated_acc)
    if current_example != "Forth/WAP" and \
       current_example != "CLUTRR":
      cols.append(group[group["Experiment"] == "C.A.E.I., A. L. N."]["Accuracy"].iloc[0] - uncalibrated_acc)
    else:
      cols.append(float('nan'))
    return pd.Series(cols, ["Experiment", "U.", "C.", "C.A.E.I."])
  accuracies_table_ = accuracies_table[accuracies_table["Experiment"].str.endswith("A. L. N.")]
  accuracies_table__ = accuracies_table_.groupby(["Example"]).apply(acc_diffs_when_ln_added)
  print(accuracies_table__)
  print(accuracies_table__.describe())

  accuracy_differences_with_label_noise = {}
  for example, experiment_without_label_noise in get_iteration_product(experiments_without_label_noise):
    if example.startswith("Noisy"):
      continue 
    accuracy_without_label_noise = accuracies_table.loc[
      (accuracies_table["Example"] == example) & \
      (accuracies_table["Experiment"] == experiment_without_label_noise) 
    ]["Accuracy"].iloc[0]
    accuracy_with_label_noise = accuracies_table.loc[
      (accuracies_table["Example"] == example) & \
      (accuracies_table["Experiment"] == f"{experiment_without_label_noise}, A. L. N.") 
    ]["Accuracy"].iloc[0]
    accuracy_differences_with_label_noise[experiment_without_label_noise] = accuracy_differences_with_label_noise.get(experiment_without_label_noise, [])
    accuracy_differences_with_label_noise[experiment_without_label_noise].append(float(accuracy_with_label_noise - accuracy_without_label_noise))
  accuracy_differences_with_label_noise = {
    k: statistics.mean(v)
    for k, v in accuracy_differences_with_label_noise.items()
  } 
  print("-- Accuracy differences with label noise:")
  print(accuracy_differences_with_label_noise)

def analyze_ECE_difference_table(ECE_difference_table):
  print("- Analyzing ECE differences table -")
  print("-- μ ECE difference by experiment")
  analysis_result_table = ECE_difference_table.groupby(["Experiment"])[["ECE before calibration μ", "ECE after calibration μ"]].mean()
  analysis_result_table["Delta"] = analysis_result_table["ECE after calibration μ"] - analysis_result_table["ECE before calibration μ"]
  ECE_before_after_delta = analysis_result_table["Delta"].mean()
  print(analysis_result_table)
  print(f"Average overall ECE difference: {ECE_before_after_delta}")

def analyze_loss_ECE_evolution_table(loss_ECE_evolution_table):
  correlation_tables = []

  print("- Analyzing loss ECE evolution table -")
  print("-- General μ ECE and cross-entropy loss correlation")
  pearson_correlation = loss_ECE_evolution_table["Cross-entropy loss"].corr(loss_ECE_evolution_table["ECE μ"], method = "pearson")
  print(pearson_correlation)

  print("-- Example-specific μ ECE and cross-entropy loss correlation")
  def get_group_ECE_loss_correlation(group):
    pearson_correlation = group["Cross-entropy loss"].corr(group["ECE μ"], method = "pearson") 
    return pd.Series([pearson_correlation], index = ["Pearson Correlation"])
  correlation_tables.append(loss_ECE_evolution_table.groupby(["Example"]).apply(get_group_ECE_loss_correlation))

  print("-- Experiment-specific μ ECE and cross-entropy loss correlation")
  def get_group_ECE_loss_correlation(group):
    pearson_correlation = group["Cross-entropy loss"].corr(group["ECE μ"], method = "pearson") 
    return pd.Series([pearson_correlation], index = ["Pearson Correlation"])
  correlation_tables.append(loss_ECE_evolution_table.groupby(["Experiment"]).apply(get_group_ECE_loss_correlation))

  for correlation_table in correlation_tables:
    print(correlation_table)

  return correlation_tables