import streamlit as st

import pandas as pd

import plotly.express as px 


data = pd.read_csv("/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Clean_df.csv")

st.dataframe(data=data)