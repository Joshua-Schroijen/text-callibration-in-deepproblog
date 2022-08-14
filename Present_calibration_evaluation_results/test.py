import pandas as pd

def row_with_max_speed(group):
  return group.iloc[group["Max Speed"].argmax()]

if __name__ == "__main__":
  f = pd.DataFrame(
    {
      'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
      'Max Speed': [380., 370., 24., 26.],
      "Min Speed": ['a', 'b', 'c', 'd']
    }
  )



  g = f.groupby(['Animal']).apply(row_with_max_speed) #max(numeric_only = True)
  print("-" * 20)
  print(g)
  print("-" * 20)
  print(g.index)
  print("-" * 20)
  print(type(g))