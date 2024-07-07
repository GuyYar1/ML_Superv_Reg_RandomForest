1. Currently not fully supported. need to add the logic of column eng and RMSE and stack of hyper parameters.  config file is ready to use, but not fully ready with Hyper param.
Instead of using git action i will use a database: Firebase Realtime Database: A NoSQL cloud database that stores and syncs data in real-time across all clients.
please 



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
