import inline as inline
import matplotlib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

train_data = pd.read_csv('./ca_train.csv', index_col='Id')
test_data = pd.read_csv('./ca_test.csv', index_col='Id')

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

numerical_cols = [cname for cname in train_data.columns
                   if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numerical_cols].copy()
X_test = train_data[numerical_cols].copy()

model = RandomForestRegressor(n_estimators=50, random_state=0)
my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', model)
])

scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())

def get_score(n_estimators):
    md = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', md)
    ])

    scores = -1 * cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')
    return scores.mean()

results = {}
for index in range(50, 401, 50):
    results[index] = get_score(index)