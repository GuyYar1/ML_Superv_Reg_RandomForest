# Library
import os
from datetime import datetime
from io import StringIO

import firebase_admin
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
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
from firebase_admin import credentials
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.preprocessing import PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model on the training set
    print("_______Perform fit to learn from X train and y train______")
    print(X_train)
    print(y_train)

    print(" start model.fit ")

    model.fit(X_train, y_train)
    print(" End model.fit ")
    # Get the best hyperparameters
    # best_params = model.best_params_
    # print("Best hyperparameters:", best_params)

    # Evaluate the model on the testing set
    # Access model attributes
    feature_importances = model.feature_importances_
    feature_names = model.feature_names_in_
    print("-----------feature_importances-----------------")
    # Print the results
    print(pd.Series(feature_importances, index=feature_names).sort_values(ascending=False))
    print("________________________________")
    return X_train, X_test, y_train, y_test


def trainr_pca(model, X, y):
    # Split the data into training and testing sets
    pca = PCA(n_components=4)  # Keep 2 components
    print("_____CREATE  train_test_split USING TEST SIZE, with random tree state")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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


def predict_y(model, X_train, X_test, y_train, y_test, pca, ver=1.0, subModel=1):
    # Make predictions on the training and testing sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    a, train_rmse, test_rmse, raw_train_std, train_pred_std, test_std, xx = model_summary(
        pca, y_test, y_test_pred, y_train, y_train_pred, subModel)

    a = float((get_int_from_ini('TRAIN', 'max_depth')))

    dict_to_db = {
        "user": os.getlogin(),
        "submodel": subModel,
        "Ver": ver,
        "Train without Model from raw data": a,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "raw_train_std": raw_train_std,
        "train_pred_std": train_pred_std,
        "test_std": test_std,
        "max_depth": float(get_int_from_ini('TRAIN', 'max_depth')),
        "min_samples_split": float(get_int_from_ini('TRAIN', 'min_samples_split')),
        "min_samples_leaf": float(get_int_from_ini('TRAIN', 'min_samples_leaf')),
        "n_estimators": float(get_int_from_ini('TRAIN', 'n_estimators')),
        "max_features": float(get_int_from_ini('TRAIN', 'max_features'))
    }

    # ref = create_firebase_admin()
    # #clearfromdb(ref, ['-O1CamdoXgL3sG8aOW5G', 'O1CamszdCzx6mrlEkAu', '-O1CcvUk3rZD0iYCO76-','-O1CdFe49Ye4QqY9zW3J'])
    #
    # write_and_get_db(ref, dict_to_db)
    # json_firbase = ref.get()
    # json_firbase1 = pd.DataFrame(json_firbase['messages']).transpose()
    # print(json_firbase1)

    plt.plot(xx, xx, 'r--')
    plt.xlabel('actual')
    plt.ylabel('predicted')

    return y_train_pred, y_test_pred


def create_firebase_admin():
    return db.reference()


def model_summary(pca, y_test, y_test_pred, y_train, y_train_pred, subModel):
    print("____________Learning Metric result ____________________")
    print("train data is the data that created the model.")
    print("train data is the data that i only have. test data is not static and be changed")
    print("so when i have the model already after the fit inside the train function.")
    print("I use the model to predict the y_train from X train. same for the X,y test.")
    print("the model should have simmilar residue\ error on prediction from test,train.")
    print(f"Look below: with {pca}")
    a = round(y_train.mean(), 3)
    b = round(RMSE(y_train_pred, y_train), 3)
    c = round(RMSE(y_test_pred, y_test), 3)
    d = round(y_train.std(), 3)
    e = round(y_train_pred.std(), 3)
    f = round(y_test.std(), 3)
    print(f"data was split to submodels. This is part: {subModel}")
    print("Train without Model from raw data. mean:", a)
    print("Train RMSE:", b)
    print("Test RMSE:", c)
    print("Raw Train STD", d)
    print("Test STD", f)
    print("Train_pred STD", e)
    print("Conclusions: ")
    print("Train STD Vs. Test STD:")
    print("Train RMSE Vs. Test RMSE: RMSE should be similar but here the diff is  factors ")
    print("RMSE/STD: focuses on prediction accuracy, while STD describes data variability.")
    print("RMSE: Visualizing learning curves or comparing RMSE across different models can provide insights")
    print("________________________________")
    # print("scatterplot")
    # plt.figure(figsize=(8, 6))
    # print("-------------scatterplot--------> x=y_test, y=y_test_pred --")
    # sns.scatterplot(x=y_test, y=y_test_pred)
    xx = np.linspace(y_test.min(), y_test.max(), 100)
    return a, b, c, d, e, f, xx


### Encoding
def cut_encode(df):
    # Map categorical variables to numeric values
    cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    # Apply the mapping to the 'cut' column
    df['cut_encoded'] = df['cut'].map(cut_mapping)
    # Drop the original 'cut' column
    df = df.drop('cut', axis=1)
    return df


### ###  ###

def eda_analysis(df, learn_column, categ_heu, full=False):
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
    print("Look here")
    print(
        "-------------pairplot--------> show a plot of mix numeric values, can use hue as category distribution--------")
    sns.pairplot(df)
    plt.show(block=True)  # Display the plot

    #sns.pairplot(df, hue=categ_heu)  # show a plot of mix numeric values, can use hue as category distribution
    #plt.show()  # Display the plot

    print("-------------displot--------> visualize the distribution of tip amounts. kernel density estimate--------")
    sns.displot(data=df, x=learn_column, kde=True)  # visualize the distribution of tip amounts. kernel density estimate
    plt.show(block=True)  # Display the plot
    sns.displot(data=df, x='ModelID', kde=True)  # visualize the distribution of tip amounts. kernel density estimate
    plt.show(block=True)  # Display the plot
    print(
        "-------------df.value_counts--------> for each column show you the distribution.  text and figure bar histogram--")

    for col in df.columns:
        print(df[col].value_counts())  # Show the distribution for each column
        print()
        sns.displot(data=df, x=col, kde=True)
        plt.title(f'Bar Histogram for {col}')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
        plt.show()
        print("________________________________")


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


def impute_nan(df, method='mean'):
    """
    Impute NaN values in a DataFrame based on the specified method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        method (str): Imputation method ('mean', 'median', or 'constant').

    Returns:
        pd.DataFrame: DataFrame with NaN values imputed.
    """
    if method == 'mean':
        imputed_values = df.mean()
    elif method == 'median':
        imputed_values = df.median()
    elif method == 'constant':
        imputed_values = 999  # Replace NaNs with a constant value (adjust as needed)
    else:
        raise ValueError("Invalid imputation method. Choose 'mean', 'median', or 'constant'.")

    df.fillna(imputed_values, inplace=True)
    return df


def prepare_data(dftrain, exe_missing=False, exe_nonnumeric_code=False, exe_exclusenonnumeric=False,
                 exe_dropna=False, exe_dummies=False, exe_FromfilterYear=1001, print_info=False,
                 SubModelPerCat='ProductGroupDesc', mode="validation", df_valid=None):
    """
    Prepare data by  handling missing values, converting non-numeric columns, excluding non-numeric columns,
    dropping rows with missing values, and creating one-hot encoded columns.

    Args:
        dftrain (pd.DataFrame): Input DataFrame.
        exe_missing (bool): Execute missing value handling.
        exe_nonnumeric_code (bool): Execute non-numeric column conversion to codes.
        exe_exclusenonnumeric (bool): Execute exclusion of non-numeric columns.
        exe_dropna (bool): Execute dropping rows with missing values.
        exe_dummies (bool): Execute one-hot encoding.
        print_info (bool): Print DataFrame info at each step.
        exe_FromfilterYear: start from which year to filter . predict future so need more releavnt data
        SubModelPerCat: keep
    Returns:
        pd.DataFrame: Processed DataFrame.
    """

    if mode == 'validation':
        df = df_valid
        df_other = dftrain
    else:
        df = dftrain
        df_other = dftrain

    # Assuming 'df' is your DataFrame
    non_numeric_columns = df.select_dtypes(exclude='number').columns
    df[non_numeric_columns] = df[non_numeric_columns].apply(lambda x: x.str.lower().str.strip())

    df_orig = df.copy()
    proddef = extract_column(df_orig, 'ProductGroupDesc')  # consider to remove - fix the bug check later.
    print(df_orig.head().T)

    if exe_FromfilterYear > 0:  #& mode != "validation"
        print("filter Year  DataFrame from:", exe_FromfilterYear)
        df = df[df['YearMade'] > exe_FromfilterYear]

    if print_info:
        print("Original DataFrame info:")
        print(df.info())

    if exe_missing:
        print("exe_missing")
        # this logic can enters 0 on row for the category what means that it will fuck the data.
        df = handle_missing_values(df, mode, df_other, action='impute')  # it was IMPUTE and works but still
        if print_info:
            print("# Check to see how many examples were missing in `auctioneerID` column")
            print(df.value_counts())

    if exe_nonnumeric_code:
        # Create a copy of the DataFrame to avoid modifying the original
        # Convert non-numeric columns to codes
        for column in df.select_dtypes(exclude=['number']).columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                df[column] = pd.Categorical(df[column]).codes + 1

        # Optionally, print information about the DataFrame
        if print_info:
            print("After converting non-numeric columns to codes:")
            print(df.info())

    if exe_dummies:
        print("exe_dummies")  # one-hot encoded
        # Consider not using this with random forests
        df = pd.get_dummies(df)  # Converts categorical variables into numerical representations
        if print_info:
            print("After creating one-hot encoded columns:")
            print(df.value_counts())

    if exe_exclusenonnumeric:
        print("exe_exclusenonnumeric")
        df = df.select_dtypes(include='number')
        if print_info:
            print("if i ran exe_nonnumeric_code before so this should not have work to do")
            print("After excluding non-numeric columns:")
            print(df.value_counts())

    if exe_dropna & (mode != "validation"):
        print("exe_dropna")
        df = df.dropna()  # Remove rows with missing values
        if print_info:
            print("After dropping rows with missing values:")
            print("df_tmp.isna().sum()", df.isna().sum())

    df = concatenate_dfs(df, proddef, SubModelPerCat)  # consider to remove - fix the bug check later.

    return df


def handle_missing_values(df, mode="validation", df_other=None, action='impute'):
    """
    Handles missing values (NaNs) in a DataFrame.



    Args:
        df (pd.DataFrame): The input DataFrame.
        action (str, optional): Action to perform ('impute' or 'missing_category').
            Defaults to 'impute'.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    # in ML when i work in mode == 'validation' i need to look into the data of the dftrain and took the  mean group
    # or most frequent value of the group  and set this value on the valid df.

    # if mode == 'validation':
    #     df = df_valid
    #     df_other = dftrain
    # else:
    #     df = dftrain
    #     df_other = dftrain

    total_nan_countbefore = df.isna().sum().sum()
    print(" total_nan_count before  impute - it should be greater num", total_nan_countbefore)

    if action == 'impute':
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns

        # Initialize the iterative imputer for numeric columns
        imputer = IterativeImputer(max_iter=10, random_state=0)

        # Fit and transform the numeric columns - df other is in validation dftrain. in train is the dftrain
        # act differently if the mode is validation.  but i set the value of df_other to be tdtrain on both modes.
        # df is df_valid in mode validation and otherwise df is dftrain.
        df_imputed_numeric = pd.DataFrame(imputer.fit_transform(df_other[numeric_cols]), columns=numeric_cols)

        # Calculate mean for each numeric column (excluding 'ModelID')
        group_means = df_imputed_numeric.drop(columns=['ModelID']).mean()

        # Impute missing numeric values using group-wise means
        for col in numeric_cols:
            if col != 'ModelID':  # Exclude 'ModelID' itself
                df_imputed_numeric.loc[df_imputed_numeric[col].isna(), col] = df_imputed_numeric.loc[
                    df_imputed_numeric[col].isna(), 'ModelID'].map(group_means[col])

        # Impute categorical columns with most frequent value based on 'ProductGroupDesc'
        for col in categorical_cols:
            if col != 'ProductGroupDesc':  # Exclude 'ProductGroupDesc' itself
                most_frequent_value = df_other.groupby('ProductGroupDesc')[col].value_counts().idxmax()
                df.loc[:, col] = most_frequent_value[1]  #  Assign directly to the column
        # Combine the imputed numeric and categorical columns
        df_imputed = pd.concat([df_imputed_numeric, df[categorical_cols]], axis=1)

    elif action == 'missing_category':
        # Treat missing values as -1 (for all columns)
        df_imputed = df.fillna(-1)

    else:
        raise ValueError("Invalid action. Choose 'impute' or 'missing_category'.")

    total_nan_countafter = df_imputed.isna().sum().sum()
    print(" total_nan_count- it should be lower num, due to the imputation", total_nan_countafter)
    print(" we handled cases of imputation", (total_nan_countbefore - total_nan_countafter))
    return df_imputed


def clean_sigma_log(df, learn_column, clearedcolumn, cnt_std=3, method='sigma', column_with_long_tail='carat',
                    mode="Validation"):
    """
      gENERAL TO BOTH : df, learn_column , method
      sigma= clearedcolumn , cnt_std
      log = column_with_long_tail
    """
    # Calculate mean and standard deviation for 'carat'

    if mode == "Validation":
        return df

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
    print("build_model")
    X = df.drop(columns=learn_column)  # these are our "features" that we use to predict from
    y = df[learn_column]  # this is what we want to learn to predict

    if pca:
        X_train, X_test, y_train, y_test = trainr_pca(rf_model, X, y)
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train(rf_model, X, y)
        return X_train, X_test, y_train, y_test


def columns_to_drop(X, skip=True, learn_column=None):
    if skip:
        return X

    columntodrop = ['MachineID', 'auctioneerID', 'Backhoe_Mounting', 'Hydraulics', 'Pushblock',
                    'Ripper',
                    'Scarifier',
                    'Tip_Control',
                    'Tire_Size',
                    'Coupler_System',
                    'Grouser_Tracks',
                    'Hydraulics_Flow',
                    'Undercarriage_Pad_Width',
                    'Stick_Length',
                    'Thumb',
                    'Pattern_Changer',
                    'Grouser_Type']
    check_col_exists_df(X, columntodrop)

    X = X.drop(columns=columntodrop)
    return X


def check_col_exists_df(X, columntodrop):
    try:
        # Check if each column exists in the DataFrame
        for col in columntodrop:
            if col not in X.columns:
                print(f"Column '{col}' doesn't exist in the DataFrame.")
    except Exception as e:
        print(f"An error occurred: {e}")


def ColumnsToKeep(X, skip=True, learn_column=None):
    print("column to Keep")
    if skip:
        return X

    # Assuming X is your DataFrame
    columns_to_keep1 = ['SalesID', 'YearMade', 'range_min', 'ModelID', 'HandNum', 'saleYear_y', 'saleMonth', 'saleDay',
                        'saleDayofweek', 'saleDayofyear', 'ProductGroupDesc']
    columns_to_keep2 = ['SalesID', 'YearMade', 'range_min', 'ProductGroupDesc', 'HandNum', 'saleYear_y', 'saleMonth',
                        'saleDay',
                        'saleDayofweek', 'saleDayofyear', 'ModelID',
                        'datasource', 'auctioneerID', 'MachineHoursCurrentMeter',
                        'UsageBand', 'fiModelDesc',
                        'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb', 'Pattern_Changer',
                        'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls',
                        'Differential_Type', 'Steering_Controls']

    columns_to_keep = columns_to_keep1

    #, 'InteractionFeature', 'Decade', 'LogMachineHours']

    if not (learn_column is None):
        columns_to_keep.append(learn_column)

    # Alternatively, you can use the drop method to achieve the same result
    X = X.drop(columns=[col for col in X.columns if col not in columns_to_keep])
    return X


def predict_with_model(X_train, X_test, y_train, y_test, rf_model, ver, pca, subModel):
    if pca:
        pass  # TBD
    else:
        return predict_y(rf_model, X_train, X_test, y_train, y_test, pca, ver, subModel)


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
        sample_df = shuffled_df.head(600000)
        print('SampleFromDftrain', sample_df.shape)
    return sample_df


def firebase_init():
    # Replace 'path/to/your/serviceAccountKey.json' with the actual path to your JSON file
    cred = credentials.Certificate(
        'C:/Users/DELL/Documents/GitHub/ML_Superv_Reg_RandomForest/db17-22f40-firebase-adminsdk-6ko5w-986a994da9.json')
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://db17-22f40-default-rtdb.firebaseio.com'})


def write_and_get_db(iref, dict):
    # Create a reference to the root of the database
    # Write a string to the 'messages' node
    iref.child('messages').push(dict)
    data = iref.get()
    print("Data retrieved:", data)


# def clearfromdb(iref, keys):
#     key1 = '-O1CF0lqKOd6pyN2KdSB'
#     key2 = '-O1CFZJg7QtZTk9e-rel'
#
#     iref.child('messages').child(key1).remove()
#     iref.child('messages').child(key2).remove()
#
def clearfromdb(iref, keys):
    for key in keys:
        iref.child('messages').child(key).delete()


def parse_product_string(product_series):
    """
    Parses a product series where each element is a string.
    The function retrieves 4 series: Prod_type, range_min, range_max, and unit.
    It also drops the original column 'fiProductClassDesc' from the DataFrame.

    Args:
        product_series (pd.Series): The input product series.

    Returns:
        (Prod_type, range_min, range_max, unit) series
    """
    parts = product_series.str.split(" - ")
    Prod_type = parts.str[0]
    range_values = parts.str[1].str.split(" to ")
    range_min = pd.to_numeric(range_values.str[0], errors='coerce')
    range_max_help = range_values.str[1].str.split()  # Extract numeric part
    range_max = range_max_help.str[0]
    unit = range_max_help.str[1]
    return Prod_type, range_min, range_max, unit


def add_additional_columns(df, Prod_type, range_min, range_max, unit):
    """
    Adds 'Prod_type', 'range_min', 'range_max', and 'unit' columns to the DataFrame.
    Args:
        df (pd.DataFrame): The existing DataFrame.
        Prod_type (str): The value for the 'Prod_type' column.
        range_min (float): The value for the 'range_min' column.
        range_max (float): The value for the 'range_max' column.
        unit (str): The value for the 'unit' column.
    Returns:
        pd.DataFrame: The DataFrame with the additional columns.
    """
    df['Prod_type'] = Prod_type
    df['range_min'] = range_min
    df['range_max'] = range_max
    df['unit'] = unit
    return df


def split_and_create_columns(dftrain, columntoextract='fiProductClassDesc', mode="validation", df_valid=None):
    df = pd.DataFrame()

    if mode == 'validation':
        df = df_valid
    else:
        df = dftrain

    df = fiproduct_split_submodels(columntoextract, df)
    #df = handnum_feature(df)  # check if it is done in the train df -
    # Assuming you have a DataFrame 'df' and the categorical column is 'ProductGroup'

    # df = target_encode_categorical(df, cat_column='ProductGroup', target_column='SalePrice')
    # df = create_interaction_features(df)
    # df = generate_polynomial_features(df) - failed
    # df = bin_year_into_decades(df)  -failed
    # df = log_transform_machine_hours(df)

    if mode == 'validation':
        df_valid = df
        return df_valid
    else:
        dftrain = df
        return dftrain


def target_encode_categorical(df, cat_column, target_column):
    target_means = df.groupby(cat_column)[target_column].mean()
    df[f'{cat_column}_encoded'] = df[cat_column].map(target_means)
    return df


def create_interaction_features(df):
    'capture the interaction between the year of manufacture and the minimum range.'
    # Example: Multiply 'Feature1' and 'Feature2' to create a new interaction feature
    df['InteractionFeature'] = df['YearMade'] * df['range_min']
    return df


def generate_polynomial_features(df, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[['YearMade', 'range_min']])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names(['YearMade', 'range_min']))
    df = pd.concat([df, poly_df], axis=1)
    return df


def bin_year_into_decades(df):
    bins = [1980, 1990, 2000, 2010, 2020]  # Adjust the bins as needed
    labels = ['1980s', '1990s', '2000s', '2010s']
    df['Decade'] = pd.cut(df['YearMade'], bins=bins, labels=labels)
    return df


def log_transform_machine_hours(df):
    df['LogMachineHours'] = np.log1p(df['MachineHoursCurrentMeter'])
    return df


def handnum_feature(df):
    # Assuming 'df' is your DataFrame and 'Sale_Date' is a datetime column
    # Convert 'Sale_Date' to datetime if it's not already in that format
    df['saleYear_y'] = pd.to_datetime(df['saleYear_y'])

    # Group by 'MachineID' and count the number of unique sale dates
    hand_num_df = df.groupby('MachineID')['saleYear_y'].nunique().reset_index()

    # Merge the 'hand_num_df' back into the original DataFrame
    df = pd.merge(df, hand_num_df, on='MachineID', how='left')
    df.rename(columns={'Sale_Date_y': 'HandNum'}, inplace=True)

    # Display the resulting DataFrame
    return df


def fiproduct_split_submodels(columntoextract, df):
    # Assuming your original column name is 'fiProductClassDesc'
    # Convert 'fiProductClassDesc' to string type if it's not
    df[columntoextract] = df[columntoextract].astype(str).str.strip()  # Remove any leading/trailing spaces
    Prod_type, range_min, range_max, unit = None, None, None, None
    Prod_type, range_min, range_max, unit = parse_product_string(df[columntoextract])
    df = add_additional_columns(df, Prod_type, range_min, range_max, unit)
    # Assign column names based on the number of columns
    if Prod_type is not None and range_min is not None and range_max is not None and unit is not None:
        print("All variables have non-None values.")
    else:
        print("Warning: Split did not result in 4 columns. Check the delimiter in 'fiProductClassDesc'.")
    # Drop the original 'fiProductClassDesc' column
    df.drop(columns=[columntoextract], inplace=True)
    # # Concatenate the split columns back to the original DataFrame
    # df = pd.concat([df, split_columns], axis=1)
    return df


def generate_submission_csv(csv_file_path, modellist1, important_categ_column, learn_column, dftrain,
                            df_validorig):
    """                     ("valid.csv", list1, important_categ_column, learn_column, df_validorig)
    Generates a submission CSV file with predicted SalePrice based on the provided model.

    Args:
        csv_file_path (str): Path to the input CSV file.
        modellist1: model_list[str(df_name)] = df_name       model_list[[key_model_path] = rf_model
        important_categ_column:
        learn_column:
        catgeindx:
        dftrain:
        df_validorig:

    Returns:
        None (Creates a CSV file with the submission data).
    """


    # Handle the same way as you handled the train CSV data - cleaning, filling, etc.
    dftrain, dfvalid = preprocess_and_extract_features(dftrain, important_categ_column, learn_column, "validation", df_validorig)
    # got dictionary  with small DF
    df_valid = {group: sub_df for group, sub_df in dfvalid.groupby((ini_util.get_value('PREPROCESS',
                                                                                        'SubModelPerCat')))}
    df_dict = df_valid.copy()

    # # reverse
    # df_valid = {k: [v] for k, v in df_valid.items()}
    # df_valid = pd.DataFrame.from_dict(df_valid)
    # group_df = df_valid.groupby(ini_util.get_value('PREPROCESS', 'SubModelPerCat'))
    # df_reversed = group_df.apply(lambda x: x.reset_index(drop=True))

    ind = 1
    counter_total_dfs_valid = 0

    y_valid_pred_dict = {}

    for model_from_list, df_from_dict in zip(modellist1, df_dict.values()):
        print(f"List item: {model_from_list}, Dictionary item: {df_from_dict}")
        print("the assumption that first model is for cat i and df_dict is for cat i ")
        len_smal_df = len(df_from_dict)
        counter_total_dfs_valid = counter_total_dfs_valid + len_smal_df

        print(f" The amount of rows in valid is: {len_smal_df}  in small df name cat:{ind} ")
        print(f"--  {ind}  -- predict_valid started for:{learn_column}")

        df_valid = ColumnsToKeep(df_from_dict, False)
        # Extract relevant features
        X_valid = df_valid.set_index('SalesID')
        X_valid = df_valid
        # Check the shape of df_valid
        print(f"Shape of df_valid: {df_valid.shape}")
        # Check the shape of X_valid
        print(f"Shape of X_valid: {X_valid.shape}")
        file_path_jobmodel = model_from_list[str(ind)]
        # key_model_path
        y_valid_pred = predict_valid(X_valid, df_valid, file_path_jobmodel)
        submit_csv(y_valid_pred)
        print(f"counter_total_dfs_valid is {counter_total_dfs_valid} should be 11574")
        listitem = [ind, file_path_jobmodel, y_valid_pred]
        y_valid_pred_dict[ind] = listitem
        ind += 1
        print(len(df_validorig))
        print(len(y_valid_pred_dict))
    return y_valid_pred_dict, df_validorig   # check if  df_validorig or  df_valid


    # here there is no RMSE. due to that this is the real time data.
    # but i can compare to the test value.
    # Assuming you have a DataFrame 'df' with columns 'p' and 'x'
    #rmse_pandas = ((X_valid[''] - X_valid['x']) ** 2).mean() ** 0.5


def submit_csv(y_valid_pred):
    # Create a submission CSV file
    submission_filename = f'submission_{datetime.now().isoformat()}.csv'
    submission_filename = submission_filename.replace(":", "_")
    y_valid_pred.to_csv(submission_filename)
    print(f"Submission CSV file saved as '{submission_filename}'")


def predict_valid(X_valid, df_valid, model):

    # model is the path to the job saved before when it was train off
    # Later, load the model from the file
    loaded_rf = joblib.load(model)
    # Now you can use 'loaded_rf' for predictions
    y_valid_pred = loaded_rf.predict(X_valid)
    y_valid_pred = pd.Series(y_valid_pred, index=X_valid.index, name='SalePrice')
    y_valid_pred.index = df_valid['SalesID']
    return y_valid_pred


def create_date_features(dftrain, mode, df_valid=None):
    df = pd.DataFrame()

    if mode == 'validation':
        df = df_valid
    else:
        df = dftrain

    print(f"\nCreate Date Features (Operation mode: {mode})")
    df['saledate'] = pd.to_datetime(df['saledate'])

    # Add datetime parameters
    df['saleYear_y'] = df['saledate'].dt.year
    df['saleMonth'] = df['saledate'].dt.month
    df['saleDay'] = df['saledate'].dt.day
    df['saleDayofweek'] = df['saledate'].dt.dayofweek
    df['saleDayofyear'] = df['saledate'].dt.dayofyear

    # Drop original saledate
    df.drop('saledate', axis=1, inplace=True)

    if mode == 'validation':
        df_valid = df
        return df_valid
    else:
        dftrain = df
        return dftrain


def remove_unconsistent_rows(dftrain, mode, dfvalid=None):
    print("\nRemove Unconsistent Rows")
    print("Operation mode: {mode}")

    # df_Machines_Sorted = df.sort_values(by=['Machine_ID', 'Year_Made','Sale_Date'], ascending=[True, True, True])

    # -----------------------------------------------------------------------------

    # Group by 'Machine_ID' and check for uniqueness in 'COL1', 'COL2', 'COL3' etc.
    # Grouping:
    # The DataFrame df is grouped by the Machine_ID column using groupby().

    # Aggregation:
    # For each group, the agg() function is used to compute the number of unique values (nunique)
    # in COL1, COL2, and COL3.

    # The result:
    # A new DataFrame grouped where each row corresponds to a Machine_ID,
    # and the columns contain the count of unique values in each specified column.

    if 1 != 1:  # mode != 'train':
        print('Execution is not allowed in {mode} mode. allowed only in train mode')
        return (df)
    else:
        print('Removing unconsistent group rows...')

        Machine_ID_Grouped_df = df.groupby('MachineID').agg({
            'ModelID': 'nunique',
            'fiBaseModel': 'nunique',
            'YearMade': 'nunique',
            # 'Fi_Secondary_Desc'         : 'nunique',
            # 'Fi_Model_Series'           : 'nunique',
            # 'Fi_Model_Descriptor'       : 'nunique',
            # 'Product_Size'              : 'nunique',
            # 'Fi_Product_Class_Desc'     : 'nunique',
            # 'Product_Group'             : 'nunique',
        })

        # Create a mask where True indicates all values in the group are the same
        # Comparison:
        # (grouped == 1) creates a boolean DataFrame where each cell is True if the corresponding cell in grouped is equal to 1
        # (meaning all values in that column for that Machine_ID are the same).

        # Aggregation: .all(axis=1) combines the boolean values across columns.
        # For each row (i.e., for each Machine_ID), it returns True if all columns are True
        # (i.e., all specified columns have exactly one unique value in that group).

        mask = (Machine_ID_Grouped_df == 1).all(axis=1)

        # Add the mask to the original DataFrame
        # Mapping:
        # The map() function maps the boolean mask created for each Machine_ID back to the original DataFrame.
        # This adds a new column Mask to df, where each row indicates whether all specified columns have the same value for that Machine_ID.

        df['Mask'] = df['MachineID'].map(mask)

        # Count the number of True and False values in the 'Mask' column
        mask_counts = df['Mask'].value_counts()
        print("\nDelete unconsistent rows per Machine ID Grouping")
        print(f"Number of rows input df:{len(df)}")
        print(f"Number of False Bad values to FILTER: {mask_counts[False]}")
        print(f"Number of True Good values to STAY: {mask_counts[True]}")

        # Filtering Out Rows Where the Mask is False
        # Boolean Indexing:
        # This line uses boolean indexing to filter the DataFrame.
        # The condition df['Mask'] returns a boolean Series where the value is True for rows where the Mask column is True and False otherwise.

        # Filtering:
        # The DataFrame is filtered to include only the rows where the Mask column is True.
        # The result is stored in df_filtered.

        df_filtered = df[df['Mask']]

        # Drop the Mask Column
        df_filtered = df_filtered.drop(columns=['Mask'])

        print(f"Number of rows output df:{len(df_filtered)}")

        # df_filtered.head(10)
        return (df_filtered)


def cleans_tire_size(dftrain, mode, dfvalid=None):
    print(f"\nCleans_tire_size")
    print(f"Operation mode: {mode}")

    # Step 1: Remove the double quotes from the 'Tire_Size' column
    print("\nRemoving double quotes from 'Tire_Size' column...")
    df['Tire_Size'] = df['Tire_Size'].str.replace('"', '', regex=False)

    # Step 2: Replace 'None or Unspecified' with NaN
    print("\nReplacing 'None or Unspecified' with NaN in 'Tire_Size' column...")
    df['Tire_Size'] = df['Tire_Size'].replace('None or Unspecified', np.nan)

    # Step 3: Convert the 'Tire_Size' column to float
    print("\nConverting 'Tire_Size' column to float...")
    try:
        df['Tire_Size'] = df['Tire_Size'].astype(float)
    except ValueError as e:
        print(f"Error converting 'Tire_Size' to float: {e}")

    return df


def preprocess_and_extract_features(dftrain, important_categ_column, learn_column, mode="train", dfvalid=None):
    #   mode="train" for train  OR   mode="validation"  for test and Valid
    #   mode should be on train but here I do it for the test also .
    #
    ## Consider to add it on pre analysis ##
    # eda_analysis(dftrain, learn_column, important_categ_column, False)
    ### Prepare Data  ###


    ### Create extra column for each value in categorial column.###
    print(" Insert to preprocess_and_extract_features by mode:", mode)
    df = pd.DataFrame()  # this will be the chained df

    # dftrain = cleans_tire_size(dftrain, mode, dfvalid)   # error
    # dftrain = remove_unconsistent_rows(dftrain, mode, dfvalid)  # error

    df = create_date_features(dftrain, mode, dfvalid)
    # Training -> all samples up until 2011
    print("dftrain.shape", dftrain.shape)
    print(f" mode:{mode}  .Now all of our data is numeric and there are no missing values, "
          f"we should be able to build a machine"
          " learning model.", dftrain.head().T)

    df = sortdfbyyear(dftrain, mode, dfvalid)

    dftrain, dfvalid = sync_withmode_after_call(df, dftrain, dfvalid, mode)

    print("dftrain.shape", dftrain.shape)
    print(f" mode:{mode}  .Now all of our data is numeric and there are no missing values, "
          f"we should be able to build a machine"
          " learning model.", dftrain.head().T)

    #(df, exe_dropna=False, exe_dummies=False, exe_exclusenonnumeric=False, exe_missing=False,exe_nonnumeric_code=False)
    df = split_and_create_columns(dftrain, 'fiProductClassDesc', mode, dfvalid)

    dftrain, dfvalid = sync_withmode_after_call(df, dftrain, dfvalid, mode)
    # exe_missing=True, exe_nonnumeric_code=True,

    print("dftrain.shape", df.shape)
    print(f" mode:{mode}  .Now all of our data is numeric and there are no missing values, "
          f"we should be able to build a machine"
          " learning model.", df.head().T)

    #sign
    #(df, exe_missing=False, exe_nonnumeric_code=False, exe_exclusenonnumeric=False,
    #exe_dropna=False, exe_dummies=False, print_info=False)

    (exe_dropna, exe_dummies, exe_exclusenonnumeric, exe_missing, exe_nonnumeric_code,
     exe_FromfilterYear, print_info) = load_from_INI()

    submodelpercat = (ini_util.get_value('PREPROCESS', 'SubModelPerCat'))

    df = prepare_data(dftrain, exe_missing, exe_nonnumeric_code, exe_exclusenonnumeric, exe_dropna,
                           exe_dummies, exe_FromfilterYear, print_info, submodelpercat, mode, dfvalid)

    dftrain, dfvalid = sync_withmode_after_call(df, dftrain, dfvalid, mode)

    print("dftrain.shape", df.shape)
    print(f" mode:{mode}  .Now all of our data is numeric and there are no missing values, "
          f"we should be able to build a machine"
          " learning model.", df.head().T)
    # Convert all columns to int64
    # df = dftrain.astype(int) error


    if mode =="validation":
        dfvalid = aligned_columns_asdftrainmodel(dftrain, dfvalid)
        # Now df_aligned has the same columns and index as dftrain

    print("dftrain.shape", df.shape)
    print(f" mode:{mode}  .Now all of our data is numeric and there are no missing values, "
          f"we should be able to build a machine"
          " learning model.", df.head().T)


    # here the first index  is correct and its value is:  1222837  from here to the end there is a row that
    # changes the index.  look!!!
    ## sig: (df, learn_column, clearedcolumn, cnt_std=3, method='sigma', column_with_long_tail='carat', ):
    #dftrain = clean_sigma_log(dftrain, learn_column, important_categ_column, 3, 'sigma', important_categ_column, mode)
    #print("dftrain.shape", dftrain.shape)

    ## eda_analysis(dftrain, learn_column, important_categ_column, True)

    #print(dftrain[SubModelPerCat].unique())

    #check if the first index is 1222837 and not  1222843



    return dftrain, dfvalid


def sync_withmode_after_call(df, dftrain, dfvalid, mode):
    if mode == 'validation':
        dfvalid = df
        dftrain = dftrain
    else:
        dftrain = df
        dfvalid = dfvalid
    return dftrain, dfvalid


def aligned_columns_asdftrainmodel(dftrain, dfvalid):
    # Assuming dftrain and dfvalid are your DataFrames
    common_columns = dftrain.columns.intersection(dfvalid.columns)

    # Create a new DataFrame with common columns
    df_aligned = dfvalid[common_columns]

    # dftrain and dfvalid has different amount of rows so cant set index of one on another.
    # Now df_aligned has the same columns and index as dftrain
    return df_aligned


def sortdfbyyear(dftrain, mode, df_valid):
    df = pd.DataFrame()

    if mode == 'validation':
        df = df_valid
    else:
        df = dftrain

    print("sortdfbyyear - saleYear_y or saleYear_y_y")
    df.saleYear_y.value_counts().sort_index()  # check saleYear_y_y or saleYear_y

    if mode == 'validation':
        df_valid = df
        return df_valid
    else:
        dftrain = df
        return dftrain


def extract_column(df1, column_name):
    # Select the specified column from df1
    extracted_series = df1.loc[:, column_name]

    # Convert the Series to a DataFrame with the same index
    extracted_df = extracted_series.to_frame()

    extracted_df.columns = [column_name]
    return extracted_df


def concatenate_dfs(df1, extracted_df, SubModelPerCat):
    # Convert the column to categorical
    extracted_df[SubModelPerCat] = extracted_df[SubModelPerCat].astype('category').cat.codes + 1
    # Merge the dataframes
    if not (SubModelPerCat in df1.columns):
        result_df = df1.join(extracted_df)
        df1 = result_df
    return df1


def load_from_INI():
    exe_missing = get_bool_from_ini('PREPROCESS', 'exe_missing')
    exe_nonnumeric_code = get_bool_from_ini('PREPROCESS', 'exe_nonnumeric_code')
    exe_exclusenonnumeric = get_bool_from_ini('PREPROCESS', 'exe_exclusenonnumeric')
    exe_dropna = get_bool_from_ini('PREPROCESS', 'exe_dropna')
    exe_dummies = get_bool_from_ini('PREPROCESS', 'exe_dummies')
    print_info = get_bool_from_ini('PREPROCESS', 'print_info')
    exe_FromfilterYear = get_int_from_ini('PREPROCESS', 'exe_FromfilterYear')
    return (exe_dropna, exe_dummies, exe_exclusenonnumeric, exe_missing, exe_nonnumeric_code, exe_FromfilterYear,
            print_info)


def load_job():
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    # Create and train your Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=250, max_features=9)
    rf.fit(X_train, y_train)

    # Save the model to a file
    joblib.dump(rf, 'random_forest_model.joblib')

    # Later, load the model from the file
    loaded_rf = joblib.load('random_forest_model.joblib')

    # Now you can use 'loaded_rf' for predictions
    preds = loaded_rf.predict(X_test)


def createfeatureobefirst():
    from sklearn.ensemble import RandomForestRegressor

    # Assuming you have X_train and y_train
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Get feature importances
    feature_importances = rf.feature_importances_

    # Find the index of the desired feature
    desired_feature_index = feature_names.index('desired_feature')

    # Reorder features (move desired feature to the front)
    new_X_train = X_train[:,
                  [desired_feature_index] + [i for i in range(X_train.shape[1]) if i != desired_feature_index]]

    # Retrain the model with reordered features
    rf_reordered = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reordered.fit(new_X_train, y_train)


def combine_dfs(list_of_dfs):
    # Concatenate DataFrames vertically (along rows)
    combined_df = pd.concat(list_of_dfs, axis=0)

    # Remove the first row (title)
    combined_df = combined_df.iloc[1:]

    # Set the index column as 'SalesID' (assuming it's the index in each small DataFrame)
    combined_df.index.name = 'SalesID'

    # Rename the 'SalePrice' column (adjust this if needed)
    df = pd.DataFrame({'SalesID': combined_df.index.astype(int), 'SalePrice': combined_df.values})

    return combined_df


def main():
    """
    Model creation:
        A. Pre:
            1. feature eng. create columns ,exe_missing, exe_nonnumeric_code,  exe_exclusenonnumeric, exe_dropna, exe_dummies,
            2. statitcs
            3. outlier handling

        B.
            1. Model init - Hyper params

        C. Post
            1. Feature Importance, permutation,  RMSE Test\Train

     """

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
    important_categ_column = ini_util.get_value('MODULE',
                                                'important_categ_column')  # want to see different distribution on plot
    num = ini_util.get_value('TRAIN', 'random_state')
    print(f"learn_column is: {learn_column}")
    print(f"important_categ_column is: {important_categ_column}")
    print(f"num is: {num}")
    # when sns i have only train which will be later split- consider to change TODO
    #  dftrain will be split to train and Test, dfvalid available only when df comes from url due to BIG data
    en = get_bool_from_ini('DATASET', 'url_en')

    dftrain, dfvalid = get_df(en)

    dftrain.head()
    dftrain = SampleFromDftrain(dftrain, False)  # remove later

    # Instantiate the Random Forest regressor
    # rf = RandomForestRegressor(random_state=42)

    # # Define hyperparameter distributions
    # param_dist = {
    #     'n_estimators': randint(50, 200),  # Randomly sample n_estimators
    #     'max_depth': randint(5, 20),  # Randomly sample max_depth
    #     'min_samples_split': randint(2, 20)  # Randomly sample min_samples_split
    # }
    #
    # # Instantiate Randomized Search
    # rf_model = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
    #                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

    #


    #max_features=9
    # max_leaf_nodes: None (unlimited number of leaf nodes)
    # min_samples_leaf: 1 (minimum number of samples required to be at a leaf node)
    ## max_leaf_nodes min_samples_leaf
    ### EDA Exploratory  Data analysis ###

    # ************&&&&&&&&&&&&&&&&&***********************
    cleareddf, dfvalid = preprocess_and_extract_features(dftrain, important_categ_column, learn_column, "train")

    cleareddf = {group: sub_df for group, sub_df in cleareddf.groupby((ini_util.get_value('PREPROCESS',
                                                                                          'SubModelPerCat')))}
    # ************&&&&&&&&&&&&&&&&&***********************

    # Group by 'ProductGroupDesc' and create a dictionary of dataframes with their name of category not numeric

    # Initialize an empty list to store models
    model_dictt = {}
    modellist1 = []

    for df_name, df in cleareddf.items():
        tt = df.copy()
        len_smal_df = len(df)
        print(f" The amount of rows is: {len_smal_df}  in small df name cat:{df_name} ")
        print(f"look if they are ordered cat:", tt.info())
        print(f"@Cycle {df_name} for category  train creation")
        print(df_name)
        print(f" mode: validation .  look here what is the first index {df.head().T} , category:{df_name}")
        df = ColumnsToKeep(df, False, learn_column)  # skip for now

        # perform analysis before train is built next row
        #eda_analysis(df,learn_column, 'ModelID', True)

        rf_model = RandomForestRegressor(random_state=get_int_from_ini('TRAIN', 'random_state'),
                                         max_depth=get_int_from_ini('TRAIN', 'max_depth'),
                                         min_samples_split=get_int_from_ini('TRAIN', 'min_samples_split'),
                                         min_samples_leaf=get_int_from_ini('TRAIN', 'min_samples_leaf'),
                                         n_estimators=get_int_from_ini('TRAIN', 'n_estimators'),
                                         max_features=get_int_from_ini('TRAIN', 'max_features'))

        print(f" mode: validation .  look here what is the first index {df.head().T} , category:{df_name}")
        X_train, X_test, y_train, y_test = build_model(rf_model, df, learn_column, False, 1.0)

        y_train_pred, y_test_pred = predict_with_model(X_train, X_test, y_train, y_test, rf_model, 1.0, False,
                                                       df_name)

        print(f" mode: validation .  look here what is the first index {df.head().T} , category:{df_name}")
        # Save the model to a file
        filename = str(df_name) +'random_forest_model.joblib'
        joblib.dump(rf_model, filename)
        model_dictt[str(df_name)] = filename
        modellist1.append(model_dictt)
        print(f" mode: validation .  look here what is the first index {df.head().T} , category:{df_name}")

        print("--- Cycle categ  small dfs--- ")

    df_validorig = empty_df = pd.DataFrame()

    # Load the CSV file
    print("generate_submission_csv")
    project_root1 = r"C:\Users\DELL\Documents\GitHub\ML_Superv_Reg_RandomForest"
    valid_file_path1 = os.path.join(project_root1, "valid.csv")
    df_valid = pd.read_csv(valid_file_path1)  # 11574
    df_validorig = df_valid.copy()

     #y_valid_pred_dict, df_validorig
    dictofmodel_ypred , df_validorig = generate_submission_csv("valid.csv", modellist1, important_categ_column, learn_column, dftrain, df_validorig)
    # _______________________________________________________________________
    ## end ##
    # dictofmodel_ypred  . format: [ind, model_from_list, y_valid_pred]
    #                       Key[0] = [ind=0 , model_from_list for cat=0 , y_valid_pred for cat=0]
    #                      Key[1] = [ind=1 , model_from_list for cat=1 , y_valid_pred for cat=1]  till the 6 or more cat
    #  exe_Missing could cause issues in the category

    concatenated_y_valid_pred = pd.DataFrame()
    df_concatenated = None
    # Iterate through the keys
    for key, values in dictofmodel_ypred.items():
        y_valid_pred_series = values[2]
        concatenated_y_valid_pred = pd.concat([concatenated_y_valid_pred, y_valid_pred_series])
        print(len(concatenated_y_valid_pred))
        # Assuming conflattened_y_valid_pred = concatenated_y_valid_pred['y_valid_pred'].squeeze()catenated_y_valid_pred is a DataFrame with a 'y_valid_pred' column

    print(concatenated_y_valid_pred.info())

    # Rename the current index to 'SalesID'
    concatenated_y_valid_pred.index.name = 'SalesID'

    if len(concatenated_y_valid_pred.columns) > 1:
        concatenated_y_valid_pred = concatenated_y_valid_pred.rename(
            columns={concatenated_y_valid_pred.columns[1]: 'SalePrice'})
    else:
        # Handle the case where the second column is missing
        print("Warning: DataFrame does not have a second column to rename.")
        # You might want to add logic here to handle this situation
    # Reset index to make 'SalesID' a regular column
   # concatenated_y_valid_pred = concatenated_y_valid_pred.reset_index()

    # Concatenate all DataFrames vertically
    # Assuming your DataFrame is named 'combined_df'
    submit_csv(concatenated_y_valid_pred)  # Final File submit


    print("--- END Run Good Bye--- ")


if __name__ == "__main__":
    # Adjust the path based on your project location
    project_root = r"C:\Users\DELL\Documents\GitHub\ML_Superv_Reg_RandomForest"
    ini_file_name = "config.INI"
    ini_file_path = os.path.join(project_root, ini_file_name)
    # Create an instance of SingletonINIUtility
    ini_util = SingletonINIUtility(ini_file_path)
    main()
