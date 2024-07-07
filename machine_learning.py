# Library
import os
from io import StringIO

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from utils.INI_Utility import *
import gdown
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path
from firebase_admin import db


def download_from_gdrive(url, filename):
    # Extract the file ID from the URL
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Download the file
    if Path(filename).exists():
        print(f"File '{filename}' already exists. Skipping download.")
    else:
        gdown.download(download_url, filename, quiet=False)
        print(f"File downloaded as: {filename}")

def get_df_Url():

    ' the two url should be in config'
    train_url = initialize_ini().get_value('DATASET', 'train_url')
    valid_url = initialize_ini().get_value('DATASET', 'valid_url')
    # Example usage

    download_from_gdrive(train_url, 'train.csv')
    download_from_gdrive(valid_url, 'valid.csv')

    df_train = pd.read_csv('train.csv')
    df_valid = pd.read_csv('valid.csv')

    print(df_train.head())
    print(df_valid.head())

    return df_train, df_valid


def get_df_sns():
    name = initialize_ini().get_value('DATASET', 'sns_name')
    df_all = sns.load_dataset(name)  # ('tips')
    return df_all, None


def get_df(url_en):
    """
    url_en = True so retrive df from url_name
    url_en = False so retrive df from sns name using url_name
    """
    if url_en:
        return get_df_Url()
    else:
        return get_df_sns()


def initialize_ini():

    """        
    :return: 
    """""" Get value from section 'TRAIN'
    learning_rate = ini_util.get_value('TRAIN', 'url')
     Set value in section 'VALID'
    ini_util.set_value('VALID', 'url', '200')
    ini_util.save_changes()
    """
    ini_file = "config.INI"
    ini_util = SingletonINIUtility(ini_file)
    ini_util.read_ini()
    return ini_util

def train(model, X, y):
    # Split the data into training and testing sets
    print("_____CREATE  train_test_split USING TEST SIZE, with random tree state")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    # Train the model on the training set
    print("_______Perform fit to learn from X train and y train______")
    model.fit(X_train, y_train)
    # Evaluate the model on the testing set
    # Access model attributes
    # Calculate feature importances
    feature_importances = model.feature_importances_
    feature_names = model.feature_names_in_

    permotation(X, feature_importances, feature_names, model, y)

    #permutation(X, model, y,feature_importances)

    print("-----------feature_importances-----------------")
    # Print the results
    print(pd.Series(feature_importances, index=feature_names).sort_values(ascending=False))
    print("________________________________")
    return X_train, X_test, y_train, y_test


def permotation(X, feature_importances, feature_names, model, y):
    # Calculate permutation importances
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    # Add permutation importances to the DataFrame
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df['Permutation_Importance'] = result.importances_mean
    # Sort features by importance
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    # Print the sorted DataFrame
    print(feature_importance_df)


def permutation(X, model, y, feature_importance_df=None):
    # Calculate permutation importances
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    # Add permutation importances to the DataFrame
    feature_importance_df['Permutation_Importance'] = result.importances_mean
    # Sort features by importance
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    # Print the sorted DataFrame
    print(feature_importance_df)


def trainr_pca(model, X, y):
    # Split the data into training and testing sets
    pca = PCA(n_components=2)  # Keep 2 components
    print("_____CREATE  train_test_split USING TEST SIZE, with random tree state")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    X_train_reduced = pca.transform(X_train)
    pca.fit(X_train)
    X_train_reduced = pca.transform(X_train)
    X_test_reduced = pca.transform(X_test)
    # Train the model on the training set
    print("_______Perform fit to learn from X train and y train______")
    # Evaluate the model on the testing set
    # Access model attributes
    feature_importances = model.feature_importances_
    feature_names = model.feature_names_in_
    print("-----------feature_importances-----------------")
    # Print the results
    print(pd.Series(feature_importances, index=feature_names).sort_values(ascending=False))
    print("________________________________")
    return X_train, X_test, y_train, y_test



def RMSE(y_pred, y_true):
    # Calculate the root mean squared error (RMSE)
    return ((y_pred - y_true) ** 2).mean() ** 0.5


def summary(model, X_train, X_test, y_train, y_test, pca, ver=1.0):
    # Make predictions on the training and testing sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("____________Learning Metric result ____________________")
    print("train data is the data that created the model.")
    print("train data is the data that i only have. test data is not static and be changed")
    print("so when i have the model already after the fit inside the train function.")
    print("I use the model to predict the y_train from X train. same for the X,y test.")
    print("the model should have simmilar residue\ error on prediction from test,train.")
    print(f"Look below: with {pca}")
    a= round(y_train.mean(), 3)
    b= round(RMSE(y_train_pred, y_train), 3)
    c= round(RMSE(y_test_pred, y_test), 3)
    d= round(y_train.std(), 3)
    e= round(y_train_pred.std(), 3)
    f= round(y_test.std(), 3)

    print("Train without Model from raw data. mean:", a)
    print("Train RMSE:", b )
    print("Test RMSE:", c )
    print("Raw Train STD", d)
    print("Train_pred STD", e)
    print("Test STD", f )
    print("Conclusions: ")
    print("Train STD Vs. Test STD:")
    print("Train RMSE Vs. Test RMSE: RMSE should be similar but here the diff is  factors ")
    print("RMSE/STD: focuses on prediction accuracy, while STD describes data variability.")
    print("RMSE: Visualizing learning curves or comparing RMSE across different models can provide insights")
    print("________________________________")
    print("scatterplot")
    plt.figure(figsize=(8, 6))
    print("-------------scatterplot--------> x=y_test, y=y_test_pred --")
    sns.scatterplot(x=y_test, y=y_test_pred)
    xx = np.linspace(y_test.min(), y_test.max(), 100)

    dict_to_db = {
          "user": os.getlogin(),
          "Ver": ver,
          "Train without Model from raw data":a,
           "train_rmse" : b,
          "test_rmse" : c,
          "raw_train_std" : d,
          "train_pred_std" : e,
          "test_std" : f,}

    ref = db.reference()
    #clearfromdb(ref, ['-O1CamdoXgL3sG8aOW5G', 'O1CamszdCzx6mrlEkAu', '-O1CcvUk3rZD0iYCO76-','-O1CdFe49Ye4QqY9zW3J'])

    write_and_get_db(ref, dict_to_db)
    json_firbase = ref.get()
    # Set display options to show all columns
    pd.set_option("display.max_columns", None)
    json_firbase1 = pd.DataFrame(json_firbase['messages']).transpose()
    print(json_firbase1)

    plt.plot(xx, xx, 'r--')
    plt.xlabel('actual')
    plt.ylabel('predicted')

    return y_train_pred, y_test_pred


### Encoding
def cut_encode(df):
    # Map categorical variables to numeric values
    cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    # Apply the mapping to the 'cut' column
    df['cut_encoded'] = df['cut'].map(cut_mapping)
    # Drop the original 'cut' column
    df = df.drop('cut', axis=1)
    return df


def clarity_encode(df):
    # Map categorical variables to numeric values
    clarity_mapping = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
    # Apply the mapping to the 'clarity' column
    df['clarity_encoded'] = df['clarity'].map(clarity_mapping)
    # Drop the original 'clarity' column
    df = df.drop('clarity', axis=1)
    return df


def color_encode(df):
    # Map categorical variables to numeric values
    color_mapping = {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1}
    # Apply the mapping to the 'color' column
    df['color_encoded'] = df['color'].map(color_mapping)
    # Drop the original 'color' column
    df = df.drop('color', axis=1)
    return df


def map_encode_all(df):
    df = cut_encode(df)
    df = clarity_encode(df)
    df = color_encode(df)
    return df


### ###  ###

def eda_analysis(df, learn_column, categ_heu, full=False):
    # Set the maximum number of rows and columns to display
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    # 5 rows in table
    print(df.head())
    #  rangeIndex, num column, dtype(float64,category int64
    print("________________________________")
    print("-------------info--------> rangeIndex, num column, dtype(float64,category int64)--------")
    print(df.info())  # rangeIndex, num column, dtype(float64,category int64)
    #  Category means non mumeric valus. i can have numeric values in category columns - Not good)
    print("________________________________")
    print(
        "-------------describe--------> perform only for numeric values which has numeric dtype a statistical  view.--------")
    print(df.describe())  # perform only for numeric values which has numeric dtype a statistical  view.
    uniqness_name(df)

    if full:
        print("Look here")
        print(
            "-------------pairplot--------> show a plot of mix numeric values, can use hue as category distribution--------")
        sns.pairplot(df)
        plt.show(block=True)  # Display the plot

        sns.pairplot(df, hue=categ_heu)  # show a plot of mix numeric values, can use hue as category distribution
        plt.show()  # Display the plot

        print(
            "-------------displot--------> visualize the distribution of tip amounts. kernel density estimate--------")
        sns.displot(data=df, x=learn_column,
                    kde=True)  # visualize the distribution of tip amounts. kernel density estimate
        plt.show(block=True)  # Display the plot

        print(
            "-------------df.value_counts--------> for each column show you the distribution.  text and figure bar histogram--")
        for col in df.columns:
            print(df.value_counts(col))  # for each column show you the distribution.  text and figure bar histogram
            print()
            plt.figure()  # Create a new figure for each plot
            sns.countplot(data=df, x=col)
            plt.title(f'Bar Histogram for {col}')
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
            plt.show(block=True)
            plt.title(f'Bar Histogram for {col}')
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
            plt.show(block=True)
            print("________________________________")


def uniqness_name(df):
    # Get unique values for each column
    for col in df.columns:
        unique_values = df[col].unique()
        print(f"Unique values counts in column '{col}': {unique_values}")


def eda_post_analysis(y_train):
    ### a little bit more EDA
    print("____________Normal Dist____________________")
    print("how many samples are between +-std from the mean?")
    print("IF Normal Dist. so between 1 std there are 68% 0.68")
    print("IF Normal Dist. so between 2 std there are 95% 0.95")
    print("IF Normal Dist. so between 3 std there are 99.7% 0.997")
    print("____________Uniform Dist____________________")
    print("IF Uniform Dist. so between 1 std there are 50% 0.50")
    print("IF Uniform Dist. so between 2 std there are 95% 0.95")
    low = y_train.mean() - y_train.std()
    high = y_train.mean() + y_train.std()

    # Solution A
    print("According to the % of samples between +-1 std. I can decide if it is Normal Dist. or UniForm:")
    print("       -->" + str(len(y_train[(y_train >= low) & (y_train <= high)]) / len(y_train)))
    print("sns.displot helps also to decide if Normal Dist. or Uniform or other")

    # solution B
    ((y_train >= low) & (y_train <= high)).sum() / len(y_train)

    # Solution C
    ((y_train >= low) & (y_train <= high)).mean()

    # Solution D
    y_train.between(low, high).mean()

    # Solution E
    y_train[y_train.between(low, high)].mean()

    # Solution F
    y_train[(y_train >= low) & (y_train <= high)].mean()
    print("________________________________")


def prepare_data(df, exe_dropna=False, exe_dummies=False):
    df_orig = df.copy()

    if exe_dropna:
        df = df.dropna()  # remove rows with na
    if exe_dummies:
        pass #
    return df


def clean_data_retrivedsig(df, learn_column, clearedcolumn, cnt_std=3, method='sigma', column_with_long_tail='carat', ):
    """
      gENERAL TO BOTH : df, learn_column , method
      sigma= clearedcolumn , cnt_std
      log = column_with_long_tail
    """
    # Calculate mean and standard deviation for 'carat'

    if learn_column == column_with_long_tail:
        print("error: Don't transform the target variable ('price'): Focus on transforming predictor variables")
        return df

    if method == 'sigma':
        mean = df[clearedcolumn].mean()
        std = df[clearedcolumn].std()

        # Define upper and lower bounds (3 standard deviations from the mean)
        upper_bound = mean + cnt_std * std
        lower_bound = mean - cnt_std * std

        # Filter the DataFrame to keep data within the bounds
        df_filtered = df[(df[clearedcolumn] >= lower_bound) & (df[clearedcolumn] <= upper_bound)]

        print("Original DataFrame shape:", df.shape)
        print("Filtered DataFrame shape:", df_filtered.shape, "sigma\std:", cnt_std * 2)
    elif method == 'log':
        # Assuming 'df' is your DataFrame and 'column_with_long_tail' is the column you want to transform
        df['transformed' + column_with_long_tail] = np.log(df[column_with_long_tail])
        df[column_with_long_tail] = df['transformed' + column_with_long_tail]
        df = df.drop('transformed' + column_with_long_tail, axis=1)
        df_filtered = df
        print(f" column{column_with_long_tail} is after log transforming due to long right tail")
    return df_filtered


def build_model(rf_model, df, learn_column, pca, ver):
    X = df.drop(columns=learn_column)  # these are our "features" that we use to predict from
    y = df[learn_column]  # this is what we want to learn to predict

    if pca:
        X_train, X_test, y_train, y_test = trainr_pca(rf_model, X, y)
    else:
        X_train, X_test, y_train, y_test = train(rf_model, X, y)
    eda_post_analysis(y_train)
    y_train_pred, y_test_pred = summary(rf_model, X_train, X_test, y_train, y_test, pca, ver)


def get_bool_from_ini(section, key):
    return ini_util.get_value(section, key).strip().lower() == 'true'


def get_int_from_ini(section, key):
    value = ini_util.get_value(section, key)
    return int(value.strip())  # Convert to integer

def SampleFromDftrain(dftrain, skip):
    # Short time - remove later
    if skip:
        sample_df = dftrain
    else:
        shuffled_df = dftrain.sample(frac=1, random_state=42)  # Set a random seed for reproducibility
        # Select the first 1000 rows
        sample_df = shuffled_df.head(15000)
    return sample_df

def firebase_init():
    import firebase_admin
    from firebase_admin import credentials

    # Replace 'path/to/your/serviceAccountKey.json' with the actual path to your JSON file
    cred = credentials.Certificate('C:/Users/DELL/Documents/GitHub/ML_Superv_Reg_RandomForest/db17-22f40-firebase-adminsdk-6ko5w-986a994da9.json')
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://db17-22f40-default-rtdb.firebaseio.com'})

def write_and_get_db(iref, dict):
    # Create a reference to the root of the database


    # Write a string to the 'messages' node
    iref.child('messages').push(dict)
    data = iref.get()
    print("Data retrieved:", data)

def clearfromdb(iref, key):
    iref.child('messages').child(key).delete()

def clearfromdb(iref, keys):
    for key in keys:
        iref.child('messages').child(key).delete()

def main():
    #SingletonINIUtility.clear()
    firebase_init()

    # Print the full path and content
    print(f"INI File Path: {ini_file_path}")
    print("INI Content:")
    for section in ini_util.config.sections():
        print(f"[{section}]")
        for key, value in ini_util.config.items(section):
            print(f"{key} = {value}")

    learn_column = ini_util.get_value('MODULE', 'learn_column')  # 'tip' #want to learn to predict
    important_categ_column = ini_util.get_value('MODULE', 'important_categ_column')    # want to see different distribution on plot
    print(f"learn_column is: {learn_column}")
    print(f"important_categ_column is: {important_categ_column}")

    # when sns i have only train which will be later split- consider to change TODO
    #  dftrain will be split to train and Test, dfvalid available only when df comes from url due to BIG data
    en = get_bool_from_ini('DATASET', 'url_en')
    dftrain, dfvalid = get_df(en)

    dftrain = SampleFromDftrain(dftrain, False)# remove later

    rr = get_int_from_ini('TRAIN', 'random_state')


    rf_model = RandomForestRegressor(random_state =get_int_from_ini('TRAIN', 'random_state'),
                                     max_depth =get_int_from_ini('TRAIN', 'max_depth'),
                                     min_samples_split = get_int_from_ini('TRAIN', 'min_samples_split'),
                                     min_samples_leaf = get_int_from_ini('TRAIN', 'min_samples_leaf')
                                     )

    # max_leaf_nodes: None (unlimited number of leaf nodes)
    # min_samples_leaf: 1 (minimum number of samples required to be at a leaf node)

    ## max_leaf_nodes min_samples_leaf
    ### EDA Exploratory  Data analysis ###
    ## Consider to add it on pre analysis ##
    eda_analysis(dftrain, learn_column, important_categ_column, False)
    ### Prepare Data
    ## df = map_encode_all(dftrain) - need to update generic way
    # Create extra column for each value in categorial column.
    prepare_data(dftrain, True, True)
    dftrain = pd.get_dummies(dftrain)  # converts categorical variables in your DataFrame df into numerical representations using one-hot encoding
    ### build the model
    dftrain.head()
    exit()
    build_model(rf_model, dftrain, learn_column, False,  1.0)# run PCA
    dftrain.head()

    print("--------------------Second try - after cleaning----------------------")
    ## sig: (df, learn_column, clearedcolumn, cnt_std=3, method='sigma', column_with_long_tail='carat', ):
    cleareddf = clean_data_retrivedsig(dftrain, learn_column, important_categ_column, 0.5, 'sigma', important_categ_column)
    # eda_analysis(cleareddf, learn_column, important_categ_column, False)

    build_model(rf_model, cleareddf, learn_column, False, 1.0)
    dftrain.head()


    # _______________________________________________________________________
    ## end ##
    # Access model attributes and store them in a variable
    print("--- END Run Good Bye--- ")


if __name__ == "__main__":
    # Adjust the path based on your project location
    project_root = r"C:\Users\DELL\Documents\GitHub\ML_Superv_Reg_RandomForest"
    ini_file_name = "config.INI"
    ini_file_path = os.path.join(project_root, ini_file_name)
    # Create an instance of SingletonINIUtility
    ini_util = SingletonINIUtility(ini_file_path)
    main()
