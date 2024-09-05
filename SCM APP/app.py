import streamlit as st

import pandas as pd

import plotly.express as px 

from PIL import Image

from IPython import display


###################################################
# VARIABLES #
# Paths
primary_data = '/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Data/SCM_Dataset_Updated_with_Green_Logistics.xlsx'
clean_data_path = '/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Data/Clean_df.csv'
train_path = '/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Data/Train.csv'
test_path = '/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Data/Test.csv'
# images
icon_image = Image.open('/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/SCM APP/images/Supply-Chain-Management.png')
model_process = Image.open('/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/SCM APP/Images/SMC Model-building-process .png')

# read the data 
data = pd.read_excel( primary_data, sheet_name='Sheet1')
Clean_df = pd.read_csv(clean_data_path)
Train = pd.read_csv(train_path)
Test = pd.read_csv(test_path)

# Create the metadata dictionary
columns = {
    'Company Name': {'sdtype': 'id'},
    'SCM Practices': {'sdtype': 'categorical'},
    'Technology Utilized': {'sdtype': 'categorical'},
    'Supply Chain Agility': {'sdtype': 'categorical'},
    'Supply Chain Integration Level': {'sdtype': 'categorical'},
    'Supply Chain Complexity Index': {'sdtype': 'categorical'},
    'Supplier Collaboration Level': {'sdtype': 'categorical'},
    'Supplier Count': {'sdtype': 'numerical'},
    'Inventory Turnover Ratio': {'sdtype': 'numerical'},
    'Lead Time (days)': {'sdtype': 'numerical'},
    'Order Fulfillment Rate (%)': {'sdtype': 'numerical'},
    'Customer Satisfaction (%)': {'sdtype': 'numerical'},
    'Supplier Lead Time Variability (days)': {'sdtype': 'numerical'},
    'Inventory Accuracy (%)': {'sdtype': 'numerical'},
    'Transportation Cost Efficiency (%)': {'sdtype': 'numerical'},
    'Cost of Goods Sold (COGS)': {'sdtype': 'numerical'},
    'Operational Efficiency Score': {'sdtype': 'numerical'},
    'Revenue Growth Rate out of (15)': {'sdtype': 'numerical'},
    'Supply Chain Risk (%)': {'sdtype': 'numerical'},
    'Supply Chain Resilience Score': {'sdtype': 'numerical'},
    'Supplier Relationship Score': {'sdtype': 'numerical'},
    'Total Implementation Cost': {'sdtype': 'numerical'},
    'Carbon Emissions (kg CO2e)': {'sdtype': 'numerical'},
    'Recycling Rate (%)': {'sdtype': 'numerical'},
    'Use of Renewable Energy (%)': {'sdtype': 'numerical'},
    'Green Packaging Usage (%)': {'sdtype': 'numerical'}
}

# Convert the dictionary into a DataFrame
metadata_df = pd.DataFrame.from_dict(columns, orient='index')
metadata_df.reset_index(inplace=True)
metadata_df.columns = ['Column Name', 'Data Type (sdtype)']
metadata_df.reset_index()


#####################################################

# FUNCTIONS

# Function to create a DataFrame with data descriptions
def create_data_description(df):
    data_description = {
        "Description": ["Number of Rows", "Number of Columns", "Missing Values", "Duplicated Rows"],
        "Value": [
            df.shape[0],
            df.shape[1],
            df.isnull().sum().sum(),
            df.duplicated().sum()
        ]
    }

    # Add data types and memory usage
    data_description["Description"].extend(["Memory Usage (KB)"])
    data_description["Value"].extend([df.memory_usage(deep=True).sum() / 1024])

    description_df = pd.DataFrame(data_description)
    return description_df


def show_sidebar_content(content_type):
    st.sidebar.title("Sidebar Information")

    if content_type == "SCM Metadata Table":
        
        st.sidebar.write("Table 3.1: Metadata table")
            
        # Display summary statistics in the sidebar
        st.sidebar.table(metadata_df.style.hide(axis='index'))

    elif content_type == "data_description":
        # Display data description in the sidebar
        st.sidebar.write("Table 3.2: Describtion table")
        description_df = create_data_description(data)
        st.sidebar.table(description_df.style.hide(axis='index'))

# ##################################################

# Display an image from a local file

st.set_page_config(page_title="Supply Chain Risk Predictive Modeling", page_icon=icon_image)
st.title("**SMC App**")

intro, analysis , rsults = st.tabs(
    [
        "Introduction",
        "Data Analysis",
        "Results"
    ]
)


with intro:

    col1 , col2 = st.columns(2)
    # 1 Introduction
    with col1:
        st.subheader("1 Introduction")
        st.write("""Welcome to SCM APP!\nSCM APP is your all-in-one platform for delving into the essential elements of Supply Chain Management.
                This app provides a variety of topics aimed at helping you comprehend and enhance your supply chain processes. 
                In this section, we'll explore the SMS concept, outline the problem statement, review the data,
                and guide you through the model-building process.""")


       # 2 What is SCM?
        st.subheader("2 What is Supply Chain Management?")
        st.write("""Supply Chain Management (SCM) involves the oversight and management of the flow of goods and services from raw material suppliers to end consumers. 
                It encompasses the planning and management of all activities involved in sourcing, procurement, conversion, and logistics.""")

        st.subheader("3 Problem Statment")
        st.write("""In today's global economy, supply chain disruptions pose a significant threat to businesses, affecting their ability to deliver products on time and maintain profitability.
                The increasing complexity of supply chains, coupled with the growing emphasis on environmental sustainability, requires companies to carefully balance efficiency, cost, and risk.
                Despite the wealth of data available, many companies struggle to effectively assess and manage supply chain risk.""")
       
    with col2:
        st.write("""Traditional risk management approaches often fail to account for the dynamic nature of global supply chains and the diverse factors influencing risk. 
                Therefore, there is a need for a data-driven approach to predict and mitigate supply chain risk, considering both SCM practices and green logistics initiatives.""")
        
        st.subheader("3 Objective")
        st.write("""The goal of this project is to create a predictive model that can accurately estimate the risk level in a supply chain. 
                 The model's goal is to identify and quantify the elements that contribute to supply chain risk,
                     allowing organizations to reduce future interruptions and optimize their supply chain operations in advance. 
                 The goal is to deliver a robust, data-driven solution that improves supply chain decision-making by accurately predicting risk.""")
        
        #***************
        # 3 DATA
        st.subheader("3 The Data")

        st.write("""The dataset titled "SCM Dataset with Green Logistics" comprises a comprehensive collection of data from various companies,
                detailing their supply chain management (SCM) practices, performance metrics, and green logistics initiatives.
                \n* Data source: [Kaggle Platform](https://www.kaggle.com/datasets/lastman0800/supply-chain-management).
                 """)
        # 3.1 Data carde 
        st.write("##### **3.1 SCM Data Card**")

        st.write("""* **Primary Key**: ***Company Name***.\n* Metadata Specification Version: ***SINGLE_TABLE_V1***.""")

        # Hyperlink to "show" the sidebar (simulated with a button)
        st.write("* Metad Data Table:")
        if st.button("Show SCM Metadata Table"):
            show_sidebar_content("SCM Metadata Table")

        # 3.1 Data Description
        st.write("##### **3.2 Data Description**")
        st.write("The following table Table 3.2 contain more information about the data.")
        if st.button("Show Data Description table"):
            show_sidebar_content("data_description")

        
        st.subheader("3 Model-building process")
        st.write("The model-building process has eight phases, as shown in the following figure:")
        st.image( model_process, caption="Model Building Process", use_column_width=True)

        

    


# Add a hyperlink to scroll to the table
# st.markdown("[Jump to SCM Metadata Table](#scm-metadata-table)")
  

# # Create a container model-building process
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


    # elif st.session_state.page == "Data Analysis":
    #     st.title("Data Analysis")
    #     st.write("This is the data analysis page.")

    # elif st.session_state.page == "Results":
    #     st.title("Results")
    #     st.write("This is the results page.")

        # st.markdown("Metadata Table: [Jump to SCM Metadata Table](#scm-metadata-table)")
