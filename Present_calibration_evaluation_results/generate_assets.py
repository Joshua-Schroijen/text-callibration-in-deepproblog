import asset_generators as ag

def format_table(table_string, resize = False):
  if resize:
    return (
      "\\begin{table}\n\\centering\n\\resizebox{\\linewidth}{!}{\n" +
      table_string +
      "}\n\end{table}"
    )
  else:
    return (
      "\\begin{table}\n\\centering\n" +
      table_string +
      "}\n\end{table}"
    )
# TODO PLOT LOGS !!!!

if __name__ == "__main__":
  # float_format="%.2f"
  with open("accuracies_table.tex", "w", encoding = "utf-8") as atf, \
       open("ECE_difference_table.tex", "w", encoding = "utf-8") as edtf, \
       open("loss_ECE_evolution_table.tex", "w", encoding ="utf-8") as leetf: 
    accuracies_table = ag.generate_accuracies_table()
    ats = accuracies_table.to_latex(
      label = "",
      position = "",
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
      label = "",
      position = "",
      index = False
    )
    edts = format_table(edts)
    edts = edts.replace("μ", "$\mu$").replace("σ", "$\sigma$")
    edtf.write(edts)
    ag.analyze_ECE_difference_table(ECE_difference_table)
    ag.plot_ECE_difference(
      ECE_difference_table,
      "ECE_differences.png"
    )
    loss_ECE_evolution_table = ag.generate_loss_ECE_evolution_table()
    leets = loss_ECE_evolution_table.to_latex(
      label = "",
      position = "",
      index = False
    )
    leets = format_table(leets)
    leets = leets.replace("μ", "$\mu$").replace("σ", "$\sigma$")
    leetf.write(leets)
    ag.plot_loss_ECE_evolution(
      loss_ECE_evolution_table,
      "loss_ECE_evolution.png"
    )