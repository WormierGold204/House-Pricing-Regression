import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_statistics(df):
    """
    Generate basic statistics of the dataset.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    dict: A dictionary containing the statistics.
    """
    stats = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "target": df.columns[-1],  # Last column is assumed to be the target variable
        "columns": df.columns.tolist(),
        "describe": df.describe().to_html(classes="table table-striped table-sm")
    }
    return stats

def target_variable_distribution(df):
    """
    Plot the distribution of the target variable.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    str: The path to the saved image.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['SalePrice'], kde=True, color='skyblue')
    plt.title(f'Distribution of SalePrice')
    plt.xlabel('SalePrice')
    plt.ylabel('count')
    path = os.path.join("static/eda", "target_distribution.png")
    plt.savefig(path)
    plt.close()
    return path

def correlation_heatmap(df):
    """
    Plot the correlation heatmap of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    str: The path to the saved image.
    """
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Pearson Correlation Heatmap")
    path = os.path.join("static/eda", "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    return path

def plots(df):
    """
    Generate EDA plots for the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    list: A list of paths to the generated plots.
    """
    plots = []

    # Numerical Features
    # Scatter plot: SalePrice vs GrLivArea
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="GrLivArea", y="SalePrice", data=df)
    plt.title("SalePrice vs Living Area")
    path = os.path.join("static/eda", "saleprice_vs_living_area.png")
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # Boxplot: SalePrice vs OverallQual
    plt.figure(figsize=(10,6))
    sns.boxplot(x="OverallQual", y="SalePrice", data=df)
    plt.title("SalePrice vs Overall Quality")
    path = os.path.join("static/eda", "saleprice_vs_overall_quality.png")
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # Scatter: SalePrice vs TotalBsmtSF
    plt.figure(figsize=(10,6))
    sns.scatterplot(x="TotalBsmtSF", y="SalePrice", data=df)
    plt.title("SalePrice vs Total Basement Area")
    path = os.path.join("static/eda", "saleprice_vs_total_basement_area.png")
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # Barplot: SalePrice vs GarageCars
    plt.figure(figsize=(10,6))
    sns.barplot(x="GarageCars", y="SalePrice", data=df)
    plt.title("SalePrice vs Number of Garage Cars")
    path = os.path.join("static/eda", "saleprice_vs_garage_cars.png")
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # Categorical Features
    # Boxplot: SalePrice vs Neighborhood
    plt.figure(figsize=(12,8))
    sns.boxplot(x="Neighborhood", y="SalePrice", data=df)
    plt.title("Average SalePrice by Neighborhood")
    plt.xticks(rotation=45)
    path = os.path.join("static/eda", "saleprice_vs_neighborhood.png")
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # Barplot: SalePrice vs MSSubClass
    plt.figure(figsize=(10,6))
    sns.barplot(x="MSSubClass", y="SalePrice", data=df)
    plt.title("Average SalePrice by Building Class")
    plt.xticks(rotation=45)
    path = os.path.join("static/eda", "saleprice_vs_building_class.png")
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # Boxplot: SalePrice vs HouseStyle
    plt.figure(figsize=(12,8))
    sns.boxplot(x="OverallCond", y="SalePrice", data=df)
    plt.title("Average SalePrice by House Style")
    plt.xticks(rotation=45)
    path = os.path.join("static/eda", "saleprice_vs_house_style.png")
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # Barplot: SalePrice vs RoofStyle
    plt.figure(figsize=(15,6))
    sns.barplot(x="Exterior1st", y="SalePrice", data=df)
    plt.title("Average SalePrice by Roof Style")
    plt.xticks(rotation=45)
    path = os.path.join("static/eda", "saleprice_vs_roof_style.png")
    plt.savefig(path)
    plots.append(path)
    plt.close()
    
    return plots