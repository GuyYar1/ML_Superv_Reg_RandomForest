                      ***ML -SUPER REGRESSOR - RANDOM FOREST***

1.The project is a way to conduct an automation framework.
2.The data exploration -EDA should be done manualy by data message.
3.I develop an automation that perform the Random Forest ML regressor. the framework expose some parameters on INI file and some tuning parameters that came from Data eng. in the level of the drievr.
4.I would like to do refactor on another branch that took the OOP approach , this branch is POC.
5.User config hyper params by INI file and run the test. results and mid results are printed in the console.
6.The logic is to devide the data to groups and examine the STD of the vartaiables.
7.in such case i develop a mechnisim that run on a small data set and trained a model that save into a job that later on loaded to the program. there was an issue to bubble the reference to other areas  due to some issue.
8.The job let the user start from the midddle or ssave results and continue to train the model when there is drift or the model is with high RMSE.
9.I use Fire base that wrote all the data on the cloud and let us examine which settable are most important among other teams.
10. I took the entire ML -EDA content  that was learned and develop it and explore other methods that are working on the production.
11. the semantic that data  is trained data and when we use test and data we should look on the trained data and decide how to fill or handle the test \valid data.
12. model comes from traing and the trainning determind how the model knew the world of the data frame and feature. so data should be simmilar to the trained data.
13. in production we will have sample or few data to predict so we must trained and allow the model learned well and let him realy learn. RMSE Train~RMSE Test. 
14. There is no RMSE of VALID due to that valid  is not known so therere is no known value to calculate the RMSE.


Details:
config file is ready to use, but not fully ready with Hyper param.
Instead of using git action i will use a database: Firebase Realtime Database: A NoSQL cloud database that stores and syncs data in real-time across all clients.
please 

Submission csv file as limitation  of 11573. error if higher rows: "Evaluation Exception: Submission must have 11573 row"

Meeting minutes: 

0. work on Feature engeniring 
1. share git - Done 
2. Add Framework capabilities - In progress { PKL is ready for deployment, INI file )
3. cloud Virtual Machine (VM)  - @ Lior To DO




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
