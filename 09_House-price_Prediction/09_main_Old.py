import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the data
housing = pd.read_csv("housing.csv")

# 2. Create a stratified test set based on income category
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

housing = strat_train_set.copy()

# 3. Separate features and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Pipelines
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
# Constructing Full pipeline
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),
                                   ("cat", cat_pipeline, cat_attribs)])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
# Now NumPy array is ready for training
print(housing_prepared.shape)

# 7. Training the Model
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
#lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
#print(f"The root mean squared error for linear Regression is {lin_rmse}")
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(lin_rmses).describe())

# Decision Tree
dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
#dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
#print(f"The root mean squared error for Decision Tree is {dec_rmse}")
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(dec_rmses).describe())


# Random Forest
random_forest_reg = RandomForestRegressor(random_state=42)
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds = random_forest_reg.predict(housing_prepared)
#random_forest_rmse = root_mean_squared_error(housing_labels, random_forest_preds)
#print(f"The root mean squared error for Random Forest is {lin_rmse}")
random_forest_rmses = -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(random_forest_rmses).describe())

