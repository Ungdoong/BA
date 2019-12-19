from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('../input/train.csv')

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]
y = data.SalePrice

model = DecisionTreeRegressor()
model.fit(X, y)

print("First in-sample predictions:", model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

print("First in-sample predictions:", model.predict(val_X.head()))
print("Actual target values for those homes:", val_y.head().tolist())

val_predictions = model.predict(val_X)

val_mae = mean_absolute_error(val_y, val_predictions)

print(val_mae)