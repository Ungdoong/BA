import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

train_data = pd.read_csv('../input/train.csv')
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features]
y = train_data.SalePrice

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X, y)

test_data = pd.read_csv('../input/test.csv')
test_X = test_data[features]

rf_val_preds = rf_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': rf_val_preds})
output.to_csv('submission.csv', index=False)
