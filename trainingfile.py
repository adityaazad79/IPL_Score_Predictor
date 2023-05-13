# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump
import numpy as np

# Reading data from csv files
res = pd.read_csv("IPL_Matches_Result_2008_2022.csv") # Reading IPL match result data
df = pd.read_csv("IPL_Ball_by_Ball_2008_2022.csv") # Reading IPL ball by ball data

# Filtering data for only first 6 overs and first 2 innings
df = df[df["overs"] < 6] # Keeping only first 6 overs data
df = df[df["innings"] < 3] # Keeping only first 2 innings data

# Grouping ball by ball data by match ID and innings
df_grp = df.groupby(["ID", "innings"])

# Aggregating total run for each match ID and innings
df_run = pd.DataFrame(df_grp.aggregate("total_run").sum())

# Merging ball by ball data with total run data
df = pd.merge(df, df_run, how='outer', on=["ID", "innings"])

# Merging ball by ball data with match result data
final_df = pd.merge(df, res, how='outer', on=["ID"])

# Extracting stadium name from venue column
v = []
for i in final_df["Venue"]:
    d = i.split(',')
    v.append(d[0])
final_df["Stadium"] = pd.DataFrame(v)

# Saving final data to csv file
final_df.to_csv("final_df.csv")

# Adding a new empty column for Bowling Team
final_df["BowlingTeam"] = ''

# Removing rows with ID 1178424 as it was causing data inconsistency
df.drop(df.loc[df['ID'] == 1178424].index, inplace=True)

# Create BowlingTeam column
final_df["BowlingTeam"] = np.where(final_df["BattingTeam"] == final_df["Team1"],
                                           final_df["Team2"], final_df["Team1"])

# Keeping only necessary columns
final_df = final_df[["ID", "Stadium", "innings", "total_run_y",
                     "BattingTeam", "BowlingTeam"]]

# Removing duplicate rows
final_df = final_df.drop_duplicates()

# Saving final data to csv file
final_df.to_csv("final_df.csv", index=False)

# Renaming columns for better readability
final_df.rename(columns={
    "Stadium": "venue",
    "BattingTeam": "batting_team",
    "BowlingTeam": "bowling_team"
}, inplace=True)

# Shuffling rows for better training
final_df = final_df.sample(final_df.shape[0])

# Separating input features and output variable
X = final_df.drop(columns=["total_run_y", "ID"])
y = final_df['total_run_y']

# Splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Defining transformation and model pipeline
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse=False, drop='first'), ["venue", "batting_team", "bowling_team"])
], remainder='passthrough')

# Create a Pipeline object 'pipe' that sequentially applies the specified transformations to the input data
pipe = Pipeline(steps=[
    ('step1', trf),  # Apply the 'trf' ColumnTransformer object to one-hot encode categorical columns
    ('step2', StandardScaler()),  # Standardize the data by subtracting the mean and dividing by the standard deviation
    ('step3', XGBRegressor(n_estimators=1000, learning_rate=0.2, max_depth=12, random_state=1))  # Train an XGBoost regression model
])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Save the pipeline object to a file using joblib
dump(pipe, 'pipe.joblib')