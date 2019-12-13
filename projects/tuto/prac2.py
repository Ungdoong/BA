import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('./US_Accidents_May19.csv')
features = ['ID', 'Start_Time', 'End_Time', 'Description', 'City', 'State']
X = data[features]
y = data['Distance(mi)']
model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)

