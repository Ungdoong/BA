import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBReressor

X_full = pd.read_csv('./ca_train.csv', index_col='Id')
X_test_full = pd.read_csv('./ca_test.csv', index_col='Id')

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)

numerical_cols = [cname for cname in X_train_full.columns
                   if X_train_full[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X_train_full.columns
                    if X_train_full[cname].dtype == 'object'
                    and X_train_full[cname].nunique() < 10]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

my_model = XGBReressor()
my_model.fit(X_train, y_train)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))