**Project Overview**


This project aims to develop an automated machine learning (ML) framework using Python. The framework leverages INI files for configuration, job tasks, PKL files for data storage, and various datasets to create an automated ML workflow. The primary focus is on enhancing the automation framework to handle more advanced tasks. This framework can be referred to as an Automated Random Forest Ensemble Framework. 

 **# System Presentation **

    ML_Superv_Reg_RandomForest.pptx -  attched.

**# Key Components**
  
  **Data Exploration and Engineering:**
  
        Initial data exploration and feature engineering were performed manually, as documented in the notebook “2024-07-13 FE-10.ipynb”.
        The dataset used includes excavator data from Kaggle.
        Automation Framework:
        Developed an automation framework that performs Random Forest ML regression based on insights from manual research.
        The framework exposes parameters via an INI file and includes tuning parameters derived from data engineering.
        
**# Workflow:**

      The framework allows loading any X data and y target for prediction, running the entire workflow of training, testing, and validation.
      It supports additional functional features for enhanced automation.
      Explanation of the Automation Framework
**# Refactoring and OOP Approach:**

      A new branch was created to refactor the framework using Object-Oriented Programming (OOP) principles. This branch serves as a Proof of Concept (POC).
      Configuration and Execution:
      Users can configure hyperparameters via an INI file and run tests. Results and intermediate outputs are printed to the console.
**# Data Grouping and Analysis:**

      The framework divides data into groups and examines the standard deviation of variables.
      A mechanism was developed to run on a small dataset, train a model, and save it as a job for later use.
**# Job Management:**

      The job system allows users to start from the middle, save results, and continue training the model when there is drift or high RMSE.
**# Cloud Integration:**

    Firebase is used to store data on the cloud, enabling examination of important settings among different teams.
**# Continuous Development:**

    The project incorporates ML and Exploratory Data Analysis (EDA) content, exploring methods suitable for production environments.
**# Future Enhancements**

    1. Fix Bugs (known issue)
    2.Implement more advanced functional features.
    3.Improve the automation framework’s robustness and scalability.
    4.Enhance the user interface for better configuration and monitoring.
               
                      
                      
More details:              
                      ***ML -SUPER REGRESSOR - RANDOM FOREST***

1.The project is a way to conduct an automation framework with research on notebook.
2.The data exploration and Engineering were done manualy, look please on the notebook attached: "2024-07-13  FE-10.ipynb"
3.I develop an automation that perform the Random Forest ML regressor based on feedbacks from the manually research. the framework expose some parameters on INI file and some tuning parameters that came from Data eng. in the level of the drievr.
4.It allows to load any X data and y target predicted and run the whole workflow of training testing and validation. There are more functional features to implement.

**Explanation on the automation framework is here**:
  1.I would like to do refactor on another branch that took the OOP approach , this branch is POC.
  2.User config hyper params by INI file and run the test. results and mid results are printed in the console.
  3.The logic is to devide the data to groups and examine the STD of the vartaiables.
  4.in such case i develop a mechnisim that run on a small data set and trained a model that save into a job that later on loaded to the program. there was an issue to bubble the reference to other areas  due to some issue.
  5.The job let the user start from the midddle or ssave results and continue to train the model when there is drift or the model is with high RMSE.
  6.I use Fire base that wrote all the data on the cloud and let us examine which settable are most important among other teams.
  7. I took the entire ML -EDA content  that was learned and develop it and explore other methods that are working on the production.
  8. the semantic that data  is trained data and when we use test and data we should look on the trained data and decide how to fill or handle the test \valid data.
  9. model comes from traing and the trainning determind how the model knew the world of the data frame and feature. so data should be simmilar to the trained data.
  10. in production we will have sample or few data to predict so we must trained and allow the model learned well and let him realy learn. RMSE Train~RMSE Test. 
  11. There is no RMSE of VALID due to that valid  is not known so therere is no known value to calculate the RMSE.
  

Details:
config file is ready to use, but not fully ready with Hyper param.
Instead of using git action i will use a database: Firebase Realtime Database: A NoSQL cloud database that stores and syncs data in real-time across all clients.
please 
a notebook is attached to the repository. 
Submission csv file as limitation  of 11573. error if higher rows: "Evaluation Exception: Submission must have 11573 row"

Meeting minutes: 

0. work on Feature engeniring -Done
1. share git - Done 
2. Add Framework capabilities - Done.
3. cloud Virtual Machine (VM)  -On Hold




How to use cloud to train  model with larger force:
Training machine learning models in the cloud involves using cloud resources to perform the training process. Here are more detailed steps for training your random forest model in the cloud:

Choose a Cloud Platform: Select a cloud provider based on your preferences and requirements. Common options include Google Cloud, Amazon Web Services (AWS), Microsoft Azure, and others.
Set Up Your Cloud Environment:
Create an account on your chosen cloud platform.
Set up a project or workspace where you’ll perform your work.
Install any necessary command-line tools or SDKs provided by the cloud provider.
Upload Your Data:
Upload your dataset (e.g., CSV files) to the cloud. You can use cloud storage services (e.g., Google Cloud Storage, Amazon S3) for this purpose.
Make sure your data is accessible within your cloud environment.
Create a Virtual Machine (VM) or Use Cloud Notebooks:
Create a VM instance or use cloud-based notebooks (e.g., Google Colab, Jupyter Notebooks on AWS SageMaker).
Install Python, required libraries (such as pandas, scikit-learn), and any other dependencies.
Write Your Code:
Write the code for your random forest model (including the build_model function you provided).
Ensure that your code references the uploaded dataset.
Run the Training Process:
Execute your build_model function within the cloud environment.
This step will train your random forest model using cloud resources (CPU, memory, etc.).
Monitor and Optimize:
Monitor the training process for any errors or issues.
Optimize hyperparameters, adjust features, and iterate as needed.
Save the Trained Model:
After successful training, the pickle dump (as shown in your original code) will save your trained model to a file (e.g., model.pkl).
Make sure this pickle file is accessible within your cloud environment.
Deployment (Separate Step):
Deployment is a separate process from training.
Deploy your trained model using cloud services (e.g., Google Cloud ML Engine, AWS Lambda, Azure Functions).
Load the trained model from the pickle file during deployment.
Set up an API or endpoint to serve predictions using the deployed model.
Testing and Validation:
Test your deployed model with sample data to ensure it works as expected.
Validate its performance and accuracy.
Remember that each cloud platform has its own specific steps and services for training and deployment. Consult the official documentation for your chosen cloud provider to get detailed instructions tailored to their environment.
