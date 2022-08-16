import asset_generators as ag

def format_table(table_string, resize = False, resize_factor = 0.7):
  lines = table_string.splitlines()
  if resize:
    lines.insert(4, "\\resizebox{" + str(resize_factor) + "\\linewidth}{!}{")
    lines.insert(-1, "}")
  return "\n".join(lines)

if __name__ == "__main__":
  with open("accuracies_table.tex", "w", encoding = "utf-8") as atf, \
       open("ECE_difference_table.tex", "w", encoding = "utf-8") as edtf, \
       open("loss_ECE_evolution_table.tex", "w", encoding = "utf-8") as leetf, \
       open("loss_ECE_evolution_corr_by_example.tex", "w", encoding = "utf-8") as leecbexaf, \
       open("loss_ECE_evolution_corr_by_experiment.tex", "w", encoding = "utf-8") as leecbexpf: 
    accuracies_table = ag.generate_accuracies_table()
    ats = accuracies_table.to_latex(
      label = "tbl:accuracies_table",
      caption = "Accuracies of each experiment",
      index = False
    )
    atf.write(format_table(ats, resize = True))
    ag.analyze_accuracies_table(accuracies_table)
    ag.plot_accuracies(
      accuracies_table,
      "accuracies.png"
    )
    ECE_difference_table = ag.generate_ECE_difference_table()
    edts = ECE_difference_table.to_latex(
      label = "tbl:ECE_difference_table",
      caption = "Before \& after final model calibration step model NN ECE μ and σ of each experiment",
      index = False
    )
    edts = format_table(edts, resize = True, resize_factor = 1.0)
    edts = edts.replace("μ", "$\mu$").replace("σ", "$\sigma$")
    edtf.write(edts)
    ag.analyze_ECE_difference_table(ECE_difference_table)
    ag.plot_ECE_difference(
      ECE_difference_table,
      "ECE_differences.png"
    )
    loss_ECE_evolution_table = ag.generate_loss_ECE_evolution_table()
    leets = loss_ECE_evolution_table.to_latex(
      label = "tbl:loss_ECE_evolution_table",
      index = False
    )
    leets = format_table(leets)
    leets = leets.replace("μ", "$\mu$").replace("σ", "$\sigma$")
    leetf.write(leets)
    correlation_tables = ag.analyze_loss_ECE_evolution_table(loss_ECE_evolution_table)
    ag.plot_loss_ECE_evolution(
      loss_ECE_evolution_table,
      "loss_ECE_evolution.png"
    )
    correlation_tables[0].to_latex(
      leecbexaf,
      label = "tbl:loss_ECE_evolution_corr_by_example",
      caption = "Pearson correlation between model loss (cross-entropy loss) and ECE across iterations by example",
      index = True
    )
    correlation_tables[1].to_latex(
      leecbexpf,
      label = "tbl:loss_ECE_evolution_corr_by_experiment",
      caption = "Pearson correlation between model loss (cross-entropy loss) and ECE across iterations by experiment",
      index = True
    )