

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
    import os
    return mo, os, pd, plt, sns, ticker, zipfile


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


@app.cell
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
    return


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


if __name__ == "__main__":
    app.run()
