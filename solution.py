

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
    import io
    import zipfile
    import os
    return mo, os, pd, zipfile


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
    # Read training and test data
    pce_train_df = pd.read_csv("./playground-series-s5e5-files/train.csv", index_col="id")
    pce_test_df = pd.read_csv("./playground-series-s5e5-files/test.csv", index_col="id")

    mo.plain(pce_train_df)
    return


if __name__ == "__main__":
    app.run()
