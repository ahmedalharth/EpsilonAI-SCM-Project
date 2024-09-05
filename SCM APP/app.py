import streamlit as st

import pandas as pd

import plotly.express as px 

from PIL import Image

from IPython import display




################################################################
# VARIABLES #
# Paths
primary_data = '/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Data/SCM_Dataset_Updated_with_Green_Logistics.xlsx'
clean_data_path = '/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Data/Clean_df.csv'
train_path = '/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Data/Train.csv'
test_path = '/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/Data/Test.csv'
# images
icon_image = Image.open('/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/SCM APP/images/Supply-Chain-Management.png')
model_process = Image.open('/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/SCM APP/Images/SMC Model-building-process  copy.png')
SCM_image = Image.open('/Users/admin/Data science/Data science EpsilonAI/EpsilonAI-course-ML/Final Project/SCM/SCM APP/Images/Supply-Chain-Management-benefits.png')
# 

# read the data 
data = pd.read_excel( primary_data, sheet_name='Sheet1')
Clean_df = pd.read_csv(clean_data_path)
Train = pd.read_csv(train_path)
Test = pd.read_csv(test_path)
Target = "Supply Chain Risk (%)"

xgb_features = ['Technology Utilized', 'Supply Chain Integration Level', 'Supplier Collaboration Level', 'Lead Time (days)',
                 'Order Fulfillment Rate (%)', 'Supplier Lead Time Variability (days)', 'Inventory Accuracy (%)', 
                 'Transportation Cost Efficiency (%)', 'Cost of Goods Sold (COGS)', 'Operational Efficiency Score','Revenue Growth Rate out of (15)',
                 'Supply Chain Resilience Score', 'Supplier Relationship Score', 'Recycling Rate (%)', 'Use of Renewable Energy (%)']
X_train = Train[xgb_features]
y_train = Train[Target]


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


# Sample data for the DataFrame ((105543, 25), (45233, 25))
final_describtion = {
    "Final Data Shape": ["(150776, 25)"],  # Replace with actual final shape of the data
    "Train Data Shape": ["(105543, 25)"],   # Replace with actual train data shape
    "Test Data Shape": ["(45233, 25)"],    # Replace with actual test data shape
    "Target Variable": ["Supply Chain Risk (%)"],  # Define your target variable
    "Models functioned": ["Linear Reggression, XGBoost, SVM"],  # List models used
    "Best Model": ["XGBoost"],  # The best performing model
    "Evaluation Metric": ["R2_score"],  # Metrics used for evaluation
    "Objective Function": ["Minimize MSE"],  # Objective function
}

# Create a DataFrame
final_dis_info = pd.DataFrame(final_describtion)

# Create a dictionary with feature names and their definitions
feature_definitions = {
    'Technology Utilized': 'The technologies and systems implemented to manage supply chain operations.',
    'Supply Chain Integration Level': 'The extent to which different elements of the supply chain work together seamlessly.',
    'Supplier Collaboration Level': 'The degree of collaboration and cooperation between the company and its suppliers.',
    'Lead Time (days)': 'The amount of time it takes from placing an order to receiving it from the supplier.',
    'Order Fulfillment Rate (%)': 'The percentage of orders that are fulfilled correctly and on time.',
    'Supplier Lead Time Variability (days)': 'The fluctuation in the time it takes for suppliers to deliver goods.',
    'Inventory Accuracy (%)': 'The accuracy of inventory records compared to actual stock available.',
    'Transportation Cost Efficiency (%)': 'The effectiveness of transportation costs in relation to the overall supply chain.',
    'Cost of Goods Sold (COGS)': 'The total cost of producing and delivering goods sold by a company.',
    'Operational Efficiency Score': 'A score representing how efficiently a company is managing its operations.',
    'Revenue Growth Rate out of (15)': 'The growth rate of a company’s revenue, measured on a scale of 15.',
    'Supply Chain Resilience Score': 'A score representing how well the supply chain can withstand disruptions.',
    'Supplier Relationship Score': 'A score evaluating the strength and quality of relationships with suppliers.',
    'Recycling Rate (%)': 'The percentage of materials that are recycled as part of the supply chain.',
    'Use of Renewable Energy (%)': 'The percentage of energy used in the supply chain that comes from renewable sources.'
}

# Convert the dictionary to a DataFrame
var_definition = pd.DataFrame(list(feature_definitions.items()), columns=['Feature', 'Definition'])


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
    st.sidebar.subheader("Sidebar Information")

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




st.set_page_config(page_title="Supply Chain Risk Predictive Modeling", page_icon=icon_image , layout="wide")
st.title("**SMC App**")

st.sidebar.header(("Deploy"))


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
                It encompasses the planning and management of all activities involved in sourcing, procurement, conversion, and logistics.
                \n The following figure show the supply change managament concepts and technique.""")
        st.image( SCM_image, caption="Figure 1.2", use_column_width=True)
        

        st.subheader("3 Problem Statment")
        st.write("""In today's global economy, supply chain disruptions pose a significant threat to businesses, affecting their ability to deliver products on time and maintain profitability.
                The increasing complexity of supply chains, coupled with the growing emphasis on environmental sustainability, requires companies to carefully balance efficiency, cost, and risk.
                Despite the wealth of data available, many companies struggle to effectively assess and manage supply chain risk.""")
        st.write("""Traditional risk management approaches often fail to account for the dynamic nature of global supply chains and the diverse factors influencing risk. 
                Therefore, there is a need for a data-driven approach to predict and mitigate supply chain risk, considering both SCM practices and green logistics initiatives.""")
        
        st.subheader("4 Objective")
        st.write("""The goal of this project is to create a predictive model that can accurately estimate the risk level in a supply chain. 
                 The model's goal is to identify and quantify the elements that contribute to supply chain risk,
                     allowing organizations to reduce future interruptions and optimize their supply chain operations in advance. 
                 The goal is to deliver a robust, data-driven solution that improves supply chain decision-making by accurately predicting risk.""")
    with col2:
        #***************
        # 3 DATA
        st.subheader("5 The Data")

        st.write("""The dataset titled "SCM Dataset with Green Logistics" comprises a comprehensive collection of data from various companies,
                detailing their supply chain management (SCM) practices, performance metrics, and green logistics initiatives.
                \n* Data source: [Kaggle Platform](https://www.kaggle.com/datasets/lastman0800/supply-chain-management).
                """)
        # 3.1 Data carde 
        st.write("##### **5.1 SCM Data Card**")

        st.write("""* **Primary Key**: ***Company Name***.\n* Metadata Specification Version: ***SINGLE_TABLE_V1***.""")

        # Hyperlink to "show" the sidebar (simulated with a button)
        st.write("* Metad Data Table:")
        if st.button("Show SCM Metadata Table 1.5"):
            show_sidebar_content("SCM Metadata Table")

        # 3.1 Data Description
        st.write("##### **5.2 Data Description**")
        st.write("The following table contain more information about the data.")
        if st.button("Show Data Description Table 2.5"):
            show_sidebar_content("data_description")
        
        st.subheader("6 Model-building process")
        st.write("The model-building process has eight phases, as shown in the following figure:")
        st.image( model_process, caption="Figure 2.6", use_column_width=True)
        st.write("""In each step the was a challeng, as shown in Table 3.2 above the data is very small with just 1000 row!.
                However, the data contains several **informative** indicators that highlight the supply chain risk (%). So, Step 3 in Figure 1.6 **Synthetic Data for Training**
                I utilized the Gaussian Copula Synthesizer, which use traditional statistical approaches to train a model and generate synthetic data.
                Ending with 150776 rows and 25 features to go further.
                \nIn steps 4, 5, and 6, I test three distinct models: LinearRegression, SVR (Support Vector Machine Regressor), and XGBOOST Regressor.
                 each model with deffirent feature using Recursive feature elimination (RFE) which is a feature selection method that fits a model and removes 
                 the weakest feature (or features) until the specified number of features is reached.
                 Finaly the model are evaluted with differen metrics R2_Sorce and MSE (mean squre Erorr).
    
                \n***You can find the code and notebooks with details in [EpsilonAI-SCM-Project](https://github.com/ahmedalharth/EpsilonAI-SCM-Project/tree/main) repository.***""")
    # Set session state to manage tab
    st.subheader("7 Conclusion")
    st.write("""The project successfully analyzed the dataset and developed a model to predict supply chain risk. The best-performing model, XGBoost, 
    was chosen based on its superior performance across evaluation metric R2 Score. The final model provides a robust 
        and accurate solution to minimize the mean squre Erorr between the true and predicted values, enabling businesses to make proactive decisions to mitigate potential disruptions.""")
    # Display the DataFrame without index in Streamlit
    st.write("* Table 3.7: Model and Data Summary")
    st.table(final_dis_info.style.hide(axis="index"))

    st.write("""* Table 4.7: Xgboost features definition.""")


    # Display the DataFrame in Streamlit without the index
    st.write("Feature Definitions")
    st.table(var_definition.style.hide(axis="index"))

    st.write("""### In The Analysis Tab We'll explore the key imprtant Statistics concering the Supply Chain Risk(%)""")

    
# Analysis Tabe:
with analysis:
    st.title("Analysis and Instights")
    st.write("""**Welcome to SCM APP's Analysis and Insights area! In this section, we delve deeper into the data, examining significant trends, correlations, and patterns that might inform improved supply chain management decisions.
             We will investigate how various aspects such as technology utilization, supply chain integration, supplier collaboration, and other essential elements affect supply chain performance using visualizations, descriptive statistics, and sophisticated analytical tools.
              In addition, we will evaluate the risks and resilience levels, delivering practical insights to improve operations.
             By the end of this analysis, you should have a better grasp of the data, the predictive potential of key variables, and how to successfully manage supply chain risks.**""")

    st.header("**Univariate Analysis**")

    col1 , col2 = st.columns(2)

    with col1:
        st.subheader("Categorical Features")
        st.write("""* **Technology Utilized: the most technologies used by com[ony**""")
        st.table(Train["Technology Utilized"].value_counts())

    
    with col2:
        # Create bar plot with Plotly Express
        data = {
    'Technology Utilized': ['AI, Blockchain, ERP', 'AI, ERP, Robotics', 'AI, Blockchain, Robotics', 'ERP, JIT, Robotics'],
    'Count': [82404, 15279, 4093, 3767]
        }

        # Create DataFrame from the data
        df = pd.DataFrame(data)

        # Create pie chart with Plotly Express
        fig = px.pie(df, names='Technology Utilized', values='Count', title='Technology Utilized Distribution')

        # Display the pie chart in Streamlit
        st.write("## Technology Utilized Pie Chart")
        st.plotly_chart(fig)
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
