import pandas as pd

my_2darray = {"a": [1, 2, 3], "b": [4, 5, 6], "index":['z', 'x', 'y']}
df = pd.DataFrame(my_2darray)
df = df.set_index(['index'])
print(df)
print(df['b']['z'])