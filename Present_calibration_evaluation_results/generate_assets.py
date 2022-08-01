import asset_generators as ag

if __name__ == "__main__":
  df = ag.generate_accuracies_table()
  print(df)
  print("-" * 30)
  df = ag.generate_ECE_difference_table()
  print(df)