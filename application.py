import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import streamlit as st
from joblib import dump, load


pipe = load('pipe.joblib')


batting_team = ['Mumbai Indians',
                'Delhi Daredevils',
                'Royal Challengers Bangalore',
                'Rising Pune Supergiant',
                'Kolkata Knight Riders',
                'Chennai Super Kings',
                'Sunrisers Hyderabad',
                'Kings XI Punjab',
                'Rajasthan Royals',
                'Pune Warriors',
                'Lucknow Super Giants',
                'Gujarat Lions',
                'Delhi Capitals',
                'Deccan Chargers',
                'Punjab Kings',
                'Kochi Tuskers Kerala',
                'Rising Pune Supergiants',
                'Gujarat Titans']

bowling_team = batting_team

venue = ['Brabourne Stadium',
         'Subrata Roy Sahara Stadium',
         'Maharashtra Cricket Association Stadium',
         'Sawai Mansingh Stadium',
         'Feroz Shah Kotla',
         'Arun Jaitley Stadium',
         'Newlands',
         'Punjab Cricket Association Stadium',
         'Barabati Stadium',
         'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
         'Dubai International Cricket Stadium',
         'MA Chidambaram Stadium',
         'Sheikh Zayed Stadium',
         'Narendra Modi Stadium',
         'Kingsmead',
         'Eden Gardens',
         'Rajiv Gandhi International Stadium',
         'Dr DY Patil Sports Academy',
         'Wankhede Stadium',
         'Himachal Pradesh Cricket Association Stadium',
         'Sardar Patel Stadium',
         'Punjab Cricket Association IS Bindra Stadium',
         'New Wanderers Stadium',
         'Buffalo Park',
         "St George's Park",
         'SuperSport Park',
         'JSCA International Stadium Complex',
         'M.Chinnaswamy Stadium',
         'Sharjah Cricket Stadium',
         'Vidarbha Cricket Association Stadium',
         'Saurashtra Cricket Association Stadium',
         'Zayed Cricket Stadium',
         'De Beers Diamond Oval',
         'Green Park',
         'Shaheed Veer Narayan Singh International Stadium',
         'Holkar Cricket Stadium',
         'Nehru Stadium',
         'OUTsurance Oval']
innings = [2, 1]
st.title('Cricket Score Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(batting_team))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(bowling_team))

venue = st.selectbox('Select venue', sorted(venue))
innings = st.selectbox('Inning', sorted(innings))

col3, col4, col5 = st.columns(3)

if st.button('Predict Score'):

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [
                            bowling_team], 'venue': venue, 'innings': [innings]})

    result = pipe.predict(input_df)
    for i in result:
        st.header("Predicted Score after 6 overs : " +
                  str(int(i)))
