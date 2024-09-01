import streamlit as st

import pandas as pd

import plotly.express as px 



# ------------------------------------------------
data = pd.read_csv("/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Clean_df.csv")

st.dataframe(data=data)

import streamlit as st

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Introduction"

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Introduction", "Data Analysis", "Results"])

# Update session state with the selected page
if page:
    st.session_state.page = page

# Display selected page content
if st.session_state.page == "Introduction":
    st.title("Introduction")
    st.write("This is the introduction page.")


# Create a container
    with st.container():
        st.write("This is inside a container")

    # Create columns
        col1, col2 = st.columns(2)
        with col1:
            st.header("Column 1")
            st.write("Content for the first column")
        with col2:
            st.header("Column 2")
            st.write("Content for the second column")

    # Outside the container
        st.write("This is outside the container")

elif st.session_state.page == "Data Analysis":
    st.title("Data Analysis")
    st.write("This is the data analysis page.")

elif st.session_state.page == "Results":
    st.title("Results")
    st.write("This is the results page.")