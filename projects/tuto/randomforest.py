from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('./train.csv')
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]
y = data.SalePrice

train_X, val_X, train_y, val_y = train_test_split(X, y)

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)
val_preds = model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_preds)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
model.fit(train_X, train_y)
val_preds = model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_preds)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_preds = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_val_preds)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
