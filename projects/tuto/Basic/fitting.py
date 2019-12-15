from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

data = pd.read_csv('../input/train.csv')
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]
y = data.SalePrice

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max lmeaf nodes: %d \t\t Mean Absolute Error: %d"%(max_leaf_nodes, my_mae))

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

min_mae = 10 ** 20
size = 0
for max_leaf_node in candidate_max_leaf_nodes:
    val_mae = get_mae(max_leaf_node, train_X, val_X, train_y, val_y)
    if(min_mae > val_mae):
        min_mae = val_mae
        size = max_leaf_node

final_model = DecisionTreeRegressor(max_leaf_nodes=size, random_state=1)
final_model.fit(X, y)

preds = final_model.predict(val_X)
print(preds)
