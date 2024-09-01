import streamlit as st

import pandas as pd

import plotly.express as px 


data = pd.read_csv("SMC\Clean_df.csv")

st.dataframe(data=data)