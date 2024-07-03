# Library
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats


def train(model, X, y):
    # Split the data into training and testing sets
    print("_____CREATE  train_test_split USING TEST SIZE, with random tree state")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    # Train the model on the training set
    print("_______Perform fit to learn from X train and y train______")
    model.fit(X_train, y_train)
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


def summary(model, X_train, X_test, y_train, y_test):
    # Make predictions on the training and testing sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("____________Learning Metric result ____________________")
    print("train data is the data that created the model.")
    print("train data is the data that i only have. test data is not static and be changed")
    print("so when i have the model already after the fit inside the train function.")
    print("I use the model to predict the y_train from X train. same for the X,y test.")
    print("the model should have simmilar residue\ error on prediction from test,train.")
    print("Look below:")
    print("Train without Model from raw data. mean:", round(y_train.mean(), 3))
    print("Train RMSE:", round(RMSE(y_train_pred, y_train), 3))
    print("Test RMSE:", round(RMSE(y_test_pred, y_test), 3))
    print("Raw Train STD", round(y_train.std(), 3))
    print("Train_pred STD", round(y_train_pred.std(), 3))
    print("Test STD", round(y_test.std(), 3))
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

def eda_analysis(df, learn_column, categ_heu, full='false'):
    # 5 rows in table
    df.head()
    #  rangeIndex, num column, dtype(float64,category int64
    print("________________________________")
    print("-------------info--------> rangeIndex, num column, dtype(float64,category int64)--------")
    df.info()  # rangeIndex, num column, dtype(float64,category int64)
    #  Category means non mumeric valus. i can have numeric values in category columns - Not good)
    print("________________________________")
    print(
        "-------------describe--------> perform only for numeric values which has numeric dtype a statistical  view.--------")
    df.describe()  # perform only for numeric values which has numeric dtype a statistical  view.
    print("Look here")
    print(
        "-------------pairplot--------> show a plot of mix numeric values, can use hue as category distribution--------")
    sns.pairplot(df, hue=categ_heu)  # show a plot of mix numeric values, can use hue as category distribution
    print("-------------displot--------> visualize the distribution of tip amounts. kernel density estimate--------")
    sns.displot(data=df, x=learn_column, kde=True)  # visualize the distribution of tip amounts. kernel density estimate
    print(
        "-------------df.value_counts--------> for each column show you the distribution.  text and figure bar histogram--")
    if full:
        for col in df.columns:
            print(df.value_counts(col))  # for each column show you the distribution.  text and figure bar histogram
            print()
            plt.figure()  # Create a new figure for each plot
            sns.countplot(data=df, x=col)
            plt.title(f'Bar Histogram for {col}')
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
            plt.show()
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


def prepare_data(df, exe_dropna=False):
    df_orig = df.copy()

    if exe_dropna:
        df = df.dropna()  # remove rows with na


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


def build_model(rf_model, df, learn_column):
    X = df.drop(columns=learn_column)  # these are our "features" that we use to predict from
    y = df[learn_column]  # this is what we want to learn to predict
    X_train, X_test, y_train, y_test = train(rf_model, X, y)
    eda_post_analysis(y_train)
    y_train_pred, y_test_pred = summary(rf_model, X_train, X_test, y_train, y_test)


# _______________________________________________________________________
## start ##
learn_column = 'price'  # 'tip' #want to learn to predict
important_categ_column = 'carat'  # 'sex'  # want to see different distribution on plot
print(f"learn_column is: {learn_column}")
print(f"important_categ_column is: {important_categ_column}")

df = sns.load_dataset('diamonds')  # ('tips')
rf_model = RandomForestRegressor(random_state=100, max_depth=15, min_samples_split=16, min_samples_leaf=6)
# max_leaf_nodes: None (unlimited number of leaf nodes)
# min_samples_leaf: 1 (minimum number of samples required to be at a leaf node)

## max_leaf_nodes min_samples_leaf
### EDA Exploratory  Data analysis ###
## Consider to add it on pre analysis ##
#    eda_analysis(df, learn_column, important_categ_column,True)

### Prepare Data
## can't encapsulate inside a function

# df = pd.get_dummies(df) # converts categorical variables in your DataFrame df into numerical representations using one-hot encoding
df = map_encode_all(df)
# Create extra column for each value in categorial column.
prepare_data(df, True)

### build the model
build_model(rf_model, df, learn_column)
df.head()

print("--------------------Second try - after cleaning----------------------")
## sig: (df, learn_column, clearedcolumn, cnt_std=3, method='sigma', column_with_long_tail='carat', ):
cleareddf = clean_data_retrivedsig(df, learn_column, important_categ_column, 0.5, 'sigma', important_categ_column)
eda_analysis(cleareddf, learn_column, important_categ_column, True)

build_model(rf_model, cleareddf, learn_column)
df.head()

# _______________________________________________________________________
## end ##
# Access model attributes and store them in a variable
