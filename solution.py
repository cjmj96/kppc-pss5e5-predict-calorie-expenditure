

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
        # Predict Calorie Expenditure

        - [Data understanding](#data-understanding)
            - [Data collection](#data-collection)
            - [Data description](#data-description)
            - [Data exploration](#data-exploration)
            - [Data quality verification](#data-quality-verification)
        - [Data preparation](#data-preparation)
            - [Data selection](#data-seleciton)
            - [Data cleaning](#data-cleaning)
            - [Data construction](#data-construction)
            - [Data integration](#data-integration)
            - [Data formatting](#data-formatting)
        - [Modeling](#modeling)
            - [Modeling technique selection](#modeling-technique-selection)
            - [Generate test design](#generate-test-design)
            - [Model building](#model-building)
            - [Model assessment](#model-assessment)
        - [Evaluation](#evaluation)
            - [Results evaluation](#results-evaluation)
            - [Process review](#process-review)
        """
    )
    return


@app.cell
def _():
    # Import relevant packages
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import io
    import zipfile
    import random
    import os
    import lightgbm as lgb
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import root_mean_squared_log_error
    from pandas.api.types import is_numeric_dtype, is_string_dtype

    # For reproducibility
    random.seed(42)

    # Apply the default theme
    sns.set_theme(style="ticks", font_scale=0.9, rc={'figure.figsize':(5, 5)})
    sns.set_style({
        "axes.labelcolor": "gray",
        "axes.edgecolor": 'gray',
        "xtick.color": 'gray',
        "ytick.color": 'gray'
    })
    return (
        is_numeric_dtype,
        is_string_dtype,
        mo,
        mutual_info_regression,
        os,
        pd,
        plt,
        sns,
        ticker,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## **Data understanding**

        ### **Data collection**


        #### **Decompressed files**
        """
    )
    return


@app.cell
def _(os, zipfile):
    # Path to the ZIP file
    zip_file_path = './playground-series-s5e5.zip'

    # Directory where you want to extract the files
    extract_dir = 'playground-series-s5e5-files/'

    # Create the extraction directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Open the ZIP file and extract its contents
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Files successfully extracted to '{extract_dir}'")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### **Load data**""")
    return


@app.cell(hide_code=True)
def _(mo, pd):
    # Read training/testing datasets
    pce_training_df = pd.read_csv("./playground-series-s5e5-files/train.csv", index_col="id")
    pce_testing_df = pd.read_csv("./playground-series-s5e5-files/test.csv", index_col="id")

    # Show the first 5 observations in the calories burnt training/testing dataset
    mo.ui.tabs({"Training data set": mo.ui.table(pce_training_df.head()),
        "Testing data set": mo.ui.table(pce_testing_df.head())}
    )
    return pce_testing_df, pce_training_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### **Data description**

        #### **Volumetric analysis of data**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, pce_testing_df, pce_training_df, pd, plt, sns, ticker):
    # Custom formatter function
    def dynamic_axis_formatter(value, pos):
        """
        Format axis ticks:
        - <1,000: original number
        - 1,000 <= x < 1,000,000: X.XK
        - >=1,000,000: X.XM
        """
        if value < 1000:
            return f"{int(value)}"
        elif 1000 <= value < 1e6:
            formatted = f"{value/1e3:.1f}K".replace(".0K", "K")
            return formatted
        else:
            formatted = f"{value/1e6:.1f}M".replace(".0M", "M")
            return formatted

    # Identify missing values (training set)
    n_missing_values_training_df = pd.DataFrame(pce_training_df.isnull().sum(), columns=['n_miss']).sort_values(by='n_miss', ascending=False)

    # Set color palette
    sns.set_palette("pastel")
    p_training = sns.barplot(n_missing_values_training_df, x="n_miss", y=n_missing_values_training_df.index)
    # Set title
    p_training.set_title('Missing values distribution (training data set)', fontdict={'size': 14, 'color': 'grey', "position": (0.35,0)})
    # Move x label to the left
    plt.xlabel('n_miss',loc="left")
    plt.ylabel('')

    # Remove top and right axis
    sns.despine()

    # Format x axis
    p_training.xaxis.set_major_formatter(ticker.FuncFormatter(dynamic_axis_formatter))

    # Identify missing values (test set)
    n_missing_values_testing_df = pd.DataFrame(pce_testing_df.isnull().sum(), columns=['n_miss']).sort_values(by='n_miss', ascending=False)

    # Set color palette
    sns.set_palette("pastel")
    p_testing = sns.barplot(n_missing_values_testing_df, x="n_miss", y=n_missing_values_testing_df.index)
    # Set title
    p_testing.set_title('Missing values distribution (training data set)', fontdict={'size': 14, 'color': 'grey', "position": (0.35,0)})
    # Move x label to the left
    plt.xlabel('n_miss',loc="left")
    plt.ylabel('')

    # Remove top and right axis
    sns.despine()

    # Format x axis
    p_testing.xaxis.set_major_formatter(ticker.FuncFormatter(dynamic_axis_formatter))

    # Show the results in different tabs
    mo.ui.tabs({"Data dimensions (Training data set)": mo.ui.text_area(f"Observations: {pce_training_df.shape[0]:,d}, Features: {pce_training_df.shape[1]:,d}"), "Data dimensions (Testing data set)": mo.ui.text_area(f"Observations: {pce_testing_df.shape[0]:,d}, Features: {pce_testing_df.shape[1]:,d}"), "Missing values distribution (Training data set)": p_training, "Missing values distribution (Testing data set)": p_testing}
    )
    return (dynamic_axis_formatter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### **Attribute types and values**

        ##### **Attribute types**
        """
    )
    return


@app.cell
def _(pce_testing_df, pce_training_df):
    # Show data types for training/testing dataset
    print("Data types for training data")
    print(pce_training_df.dtypes.to_frame().T)
    print("Data types for testing data")
    print(pce_testing_df.dtypes.to_frame().T)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### **Attribute value ranges**""")
    return


@app.cell
def _(pce_training_df, pd):
    # Define validation rules for Pandas operations
    VALIDATION_RULES = {
        "Sex": lambda df, col: df[col].isin(['male', 'female']),
        "Age": lambda df, col: (df[col] >= 0) & (df[col] <= 125),
        "Height": lambda df, col: df[col] > 0,
        "Weight": lambda df, col: df[col] > 0,
        "Duration": lambda df, col: df[col] > 0,
        "Heart_Rate": lambda df, col: (df[col] >= 40) & (df[col] <= 190),
        "Body_Temp": lambda df, col: (df[col] >= 32) & (df[col] <= 42),
        "Calories": lambda df, col: df[col] > 0
    }

    def check_expected_ranges(df: pd.DataFrame, column: str) -> None:
        """
        Validates a DataFrame column against predefined data quality rules and prints 
        the count of valid values using Pandas operations.

        Parameters:
            df (pd.DataFrame): Input DataFrame
            column (str): Column name to validate

        Raises:
            KeyError: If specified column doesn't exist in the DataFrame
        """
        # Validate column existence
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")

        # Skip columns without defined validation rules
        if column not in VALIDATION_RULES:
            return

        # Apply validation rule and count valid entries
        validation_condition = VALIDATION_RULES[column](df, column)
        valid_count = validation_condition.sum()  # Sum of True values (True=1, False=0)

        # Print formatted results
        print(f"# of valid values in {column} column: {valid_count:,d}")

    # Process all columns with progress tracking
    for column in pce_training_df.columns:
        check_expected_ranges(pce_training_df, column)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### **Attribute relationship analysis**""")
    return


@app.cell
def _(mutual_info_regression, pce_training_df, pd, plt, sns):
    # Filter data by removing rows with at least a missing value
    pce_training_no_missing_values = pce_training_df.dropna()

    # Create copy of training data and separate features/target
    X_no_missing = pce_training_no_missing_values.copy()
    y_no_missing = X_no_missing["Calories"]  # Extract target series

    # Utility functions from Tutorial
    def make_mi_scores(X, y):
        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
        # All discrete features should now have integer dtypes
        discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores


    mi_scores = make_mi_scores(X_no_missing, y_no_missing)
    mi_scores = mi_scores.reset_index()
    mi_scores.rename(columns={'index': 'Feature', 'MI Scores': 'MI_score'}, inplace=True)
    p = sns.barplot(mi_scores, x = "MI_score", y = 'Feature')
    # Set title
    p.set_title('Multiple information scores (training set)', fontdict={'size': 14, 'color': 'grey', "position": (0.50,0)})
    # Move x label to the left
    plt.xlabel('MI score',loc="left", color="grey")
    # Move y label to the bottom
    plt.ylabel('Feature',loc="bottom", color="grey")
    # Remove top and right axis
    sns.despine()

    # Show plot
    p
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### **Features dictionary**""")
    return


@app.cell(hide_code=True)
def _(mo, pd):
    feature_data = {
        'feature': [
            'id',
            'Sex',
            'Age',
            'Height',
            'Duration',
            'Heart_Rate',
            'Body_Temp',
            'Calories'
        ],
        'meaning': [
            'Unique identifier for each individual record',
            'Biological sex of the individual (Male/Female)',
            'Age of the individual in years',
            'Height measurement in centimeters',
            'Duration of physical activity in minutes',
            'Heart rate measured in beats per minute (BPM) during activity',
            'Body temperature in Celsius during/after activity',
            'Estimated calories burned (in kilocalories) during activity'
        ]
    }

    feature_df = pd.DataFrame(feature_data)
    mo.plain(feature_df)
    return


@app.cell(hide_code=True)
def _():
    ##### **Basic statistics**
    return


@app.cell
def _(mo, pce_training_df):
    # Compute statistics for every feature
    mo.plain(pce_training_df.describe(include="all"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### **Feature distributions**""")
    return


@app.cell(hide_code=True)
def _(
    dynamic_axis_formatter,
    is_numeric_dtype,
    is_string_dtype,
    mo,
    pce_training_df,
    plt,
    sns,
    ticker,
):
    def create_feature_plot(df, column):
        """Create and return a matplotlib figure for a single feature"""
        fig, ax = plt.subplots(figsize=(10, 6))

        quantitative_columns = ["Age", "Height", "Weight", "Duration", "Heart_Rate",
                              "Body_Temp", "Calories"]
        qualitative_columns = ["Sex"]

        if (is_string_dtype(df[column])) and (column in qualitative_columns):
            if df[column].nunique() >= 7:
                sns.countplot(y=column, data=df, ax=ax)
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(dynamic_axis_formatter))
            else:
                sns.countplot(x=column, data=df, ax=ax)
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(dynamic_axis_formatter))

        elif (is_numeric_dtype(df[column])) and (column in quantitative_columns):
            sns.violinplot(x=df[column], ax=ax)

        ax.set_title(f"Distribution of {column}", fontdict={"size": 15, "color": "grey", "position": (0.15, 0)})
        # Move x label to the left
        plt.xlabel(column, loc="left", color='grey')
        plt.ylabel('# of observations', loc='bottom', color='grey')
        # Remove top and right axis
        sns.despine()
        return fig  # Return the figure object instead of showing it

    # Create a dictionary of plots
    plot_dict = {
        column: create_feature_plot(pce_training_df, column)
        for column in pce_training_df.columns
    }

    # Display in tabs - each tab will only show its own plot
    mo.ui.tabs(plot_dict)
    return


@app.cell
def _(mo):
    mo.md(r"""## **Modeling**""")
    return


@app.cell
def _(pce_training_df):
    # Select relevant features (scenario 1 with mutual information revelant features)
    relevant_features_s1 = ["Duration", "Body_Temp", "Heart_Rate"]

    # Extract predictors
    X_s1 = pce_training_df.drop("Calories", axis=1)
    print(X_s1.columns)

    # Extract target feature
    y = pce_training_df.Calories
    return


if __name__ == "__main__":
    app.run()
