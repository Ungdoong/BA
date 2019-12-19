import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

X = pd.read_csv('./ca_train.csv', index_col='Id')
X_test = pd.read_csv('./ca_test.csv', index_col='Id')

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

print("\nUnique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("Unique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())

object_cols = [col for col in X_train.columns if X_train[col].dtypes == 'object']
good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]
bad_label_cols = list(set(object_cols) - set(good_label_cols))

print('\nCategorical columns that will be label encoded:', good_label_cols)
print('Categorical columns that will be dropped from the dataset:', bad_label_cols)

label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

label_encoder = LabelEncoder()
for col in set(good_label_cols):
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.fit_transform(X_valid[col])

print("\nMAE from Approach 2 (Label Encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

high_cardinality_numcols = len([key for key in d.keys() if d[key] > 10])
num_cols_neighborhood = d['Neighborhood']

low_cardinality_cols = [key for key in d.keys() if d[key] < 10]
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

print('\nCategorical columns that will be one-hot encoded:', low_cardinality_cols)
print('Categorical columns that will be dropped from the dataset:', high_cardinality_cols)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_X_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_X_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

OH_X_cols_train.index = X_train.index
OH_X_cols_valid.index = X_valid.index

num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_X_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_X_cols_valid], axis=1)

print("\nMAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))