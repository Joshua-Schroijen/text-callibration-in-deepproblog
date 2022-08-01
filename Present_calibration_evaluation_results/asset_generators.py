import json
import os
from tkinter import N
from numpy import NaN
import pandas as pd
import seaborn as sns
import statistics

""" sns.jointplot(x='speeding', y='alcohol', data=crash_df, kind='reg')
sns.barplot(x='sex',y='total_bill',data=tips_df, estimator=np.median)
sns.countplot(x='sex',data=tips_df)
# You can separate the data into separate columns for day data
# sns.lmplot(x='total_bill', y='tip', col='sex', row='time', data=tips_df)
tips_df.head()

# Makes the fonts more readable
sns.set_context('poster', font_scale=1.4)

sns.lmplot(x='total_bill', y='tip', data=tips_df, col='day', hue='sex',
          height=8, aspect=0.6) """
"""
ZIE HIER!!!!
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({"Price": [77,76,68,70,78,79,74,75]})
df["Iterations"] = df.index
plt.figure(figsize = (15,8))
sns.lineplot(x = "Iterations", y = 'Price', data = df)
plt.savefig('foo.png')"""

res_addition_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_ff.json")
res_addition_ff_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_ff_ln.json")
res_noisy_addition_ff = os.path.join("calibration_evaluator_results", "calibration_evaluation_noisy_addition_ff.json")
res_noisy_addition_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_noisy_addition_tf.json")
res_noisy_addition_tt = os.path.join("calibration_evaluator_results", "calibration_evaluation_noisy_addition_tt.json")
res_addition_tf = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_tf.json")
res_addition_tf_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_tf_ln.json")
res_addition_tt = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_tt.json")
res_addition_tt_ln = os.path.join("calibration_evaluator_results", "calibration_evaluation_addition_tt_ln.json")
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


def generate_accuracies_table():
  accuracies_table = []
  for example_name, calibrated, calibrated_after_each_train_iteration, label_noise_added, result_file in result_objects:
    with open(result_file, "r") as result_file_fd:
      result_object = json.load(result_file_fd)
      accuracy = NaN
      if "accuracy" in result_object:
        accuracy = result_object["accuracy"]
      elif "accuracies" in result_object:
        accuracy = statistics.mean(result_object["accuracies"])
      accuracies_row = (
        example_name,
        calibrated,
        calibrated_after_each_train_iteration,
        label_noise_added,
        accuracy
      )
      accuracies_table.append(accuracies_row)

  return pd.DataFrame(
    accuracies_table,
    columns = ["Example", "Calibrated", "Calibrated after each iteration", "Label noise added", "Accuracy"]
  )

def generate_ECE_difference_table():
  ECEs_table = []
  for example_name, calibrated, calibrated_after_each_train_iteration, label_noise_added, result_file in result_objects:
    print(example_name, calibrated, calibrated_after_each_train_iteration, label_noise_added)
    if not calibrated:
      continue
    with open(result_file, "r") as result_file_fd:
      result_object = json.load(result_file_fd)
      calibration_collector = result_object["networks_evolution_collectors"]["calibration_collector"]
      before_calibration_ece_histories = calibration_collector["before_calibration_ece_history"]
      after_calibration_ece_histories = calibration_collector["after_calibration_ece_history"]
      before_last_calibration_mean_ece = statistics.mean(
        map(lambda history: history[-1], before_calibration_ece_histories.values())
      )
      before_last_calibration_stdev_ece = statistics.stdev(
        map(lambda history: history[-1], before_calibration_ece_histories.values())
      )
      after_last_calibration_mean_ece = statistics.mean(
        map(lambda history: history[-1], after_calibration_ece_histories.values())
      )
      after_last_calibration_stdev_ece = statistics.stdev(
        map(lambda history: history[-1], after_calibration_ece_histories.values())
      )

      ECEs_row = (
        example_name,
        calibrated,
        calibrated_after_each_train_iteration,
        label_noise_added,
        before_last_calibration_mean_ece,
        before_last_calibration_stdev_ece,
        after_last_calibration_mean_ece,
        after_last_calibration_stdev_ece
      )
      ECEs_table.append(ECEs_row)

  return pd.DataFrame(
    ECEs_table,
    columns = ["Example", "Calibrated", "Calibrated after each iteration", "Label noise added", "ECE before calibration μ", "ECE before calibration σ", "ECE after calibration μ", "ECE after calibration σ"]
  )