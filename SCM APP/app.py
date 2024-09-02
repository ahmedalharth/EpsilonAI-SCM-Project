import streamlit as st

import pandas as pd

import plotly.express as px 



# ------------------------------------------------
# data = pd.read_csv("/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Clean_df.csv")

# st.dataframe(data=data)
# ##################################################
# Initialize session state

# Display an image from a local file
st.image("/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/SCM APP/Supply-Chain-Management.png", caption="Supply Chain Management", use_column_width=True)


st.title("**SMC App**")

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
    st.subheader("What is Supply Chain Management?")
    st.write("""Supply Chain Management (SCM) involves the oversight and management of the flow of goods and services from raw material suppliers to end consumers. 
                It encompasses the planning and management of all activities involved in sourcing, procurement, conversion, and logistics.""")

    st.subheader("Problem Statment")
    st.write("""In today's global economy, supply chain disruptions pose a significant threat to businesses, affecting their ability to deliver products on time and maintain profitability. The increasing complexity of supply chains, coupled with the growing emphasis on environmental sustainability, requires companies to carefully balance efficiency, cost, and risk.
    Despite the wealth of data available, many companies struggle to effectively assess and manage supply chain risk. Traditional risk management approaches often fail to account for the dynamic nature of global supply chains and the diverse factors influencing risk. Therefore, there is a need for a data-driven approach to predict and mitigate supply chain risk, considering both SCM practices and green logistics initiatives.""")
    
    st.subheader("The Data")
    st.write("""SCM Dataset with Green Logistics
             The dataset titled "SCM Dataset with Green Logistics" comprises a comprehensive collection of data from various companies, detailing their supply chain management (SCM) practices, performance metrics, and green logistics initiatives.
              \n* Data source [Kaggle](https://www.kaggle.com/datasets/lastman0800/supply-chain-management).""")

# # Create a container
#     with st.container():
#         st.write("This is inside a container")

#     # Create columns
#         col1, col2 = st.columns(2)
#         with col1:
#             st.header("Column 1")
#             st.write("Content for the first column")
#         with col2:
#             st.header("Column 2")
#             st.write("Content for the second column")

#     # Outside the container
#         st.write("This is outside the container")


elif st.session_state.page == "Data Analysis":
    st.title("Data Analysis")
    st.write("This is the data analysis page.")

elif st.session_state.page == "Results":
    st.title("Results")
    st.write("This is the results page.")