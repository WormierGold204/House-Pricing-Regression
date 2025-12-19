import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_statistics(df):
    """
    Generate basic statistics of the dataset.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    
    Returns:
    dict: A dictionary containing the statistics.
    """

    missing = (
        df.isna()
        .sum()
        .to_frame(name='MissingCount')
        .assign(MissingPercent=lambda x: x['MissingCount'] / len(df))
        .query('MissingCount > 0')
        .sort_values('MissingPercent', ascending=False)
    )

    stats = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "target": df.columns[-1],
        "columns": df.columns.tolist(),
        "describe": df.describe().to_html(classes="table table-striped table-sm"),
        "missing": missing.to_html(classes="table table-striped table-sm")
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
    path = "static/eda/target_distribution.png"
    plt.savefig(path)
    plt.close()
    return path

def correlation_heatmap(df):
    """
    Plot the correlation heatmap of the DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    
    Returns:
    str: The path to the saved image.
    """
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Pearson Correlation Heatmap")
    path = "static/eda/correlation_heatmap.png"
    plt.savefig(path)
    plt.close()
    return path

def plots(df):
    """
    Generate EDA plots for the DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    
    Returns:
    list: A list of paths to the generated plots.
    """
    plots = []

    # Numerical Features
    # GrLivArea
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="GrLivArea", y="SalePrice", data=df)
    plt.title("SalePrice vs Living Area")
    path = "static/eda/saleprice_vs_living_area.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # OverallQual
    plt.figure(figsize=(10,6))
    sns.boxplot(x="OverallQual", y="SalePrice", data=df)
    plt.title("SalePrice vs Overall Quality")
    path = "static/eda/saleprice_vs_overall_quality.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # TotalBsmtSF
    plt.figure(figsize=(10,6))
    sns.scatterplot(x="TotalBsmtSF", y="SalePrice", data=df)
    plt.title("SalePrice vs Total Basement Area")
    path = "static/eda/saleprice_vs_total_basement_area.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # GarageCars
    plt.figure(figsize=(10,6))
    sns.barplot(x="GarageCars", y="SalePrice", data=df)
    plt.title("SalePrice vs Number of Garage Cars")
    path = "static/eda/saleprice_vs_garage_cars.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # GarageArea
    plt.figure(figsize=(10,6))
    sns.scatterplot(x="GarageArea", y="SalePrice", data=df)
    plt.title("SalePrice vs Area of Garage")
    path = "static/eda/saleprice_vs_garage_area.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # Derived features
    # TotalSF
    plt.figure(figsize=(12,8))
    sns.scatterplot(x="TotalSF", y="SalePrice", data=df)
    plt.title("Average SalePrice by Total Square Feet")
    plt.xticks(rotation=45)
    path = "static/eda/saleprice_vs_totalsf.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # HouseAge
    plt.figure(figsize=(12,8))
    sns.scatterplot(x="HouseAge", y="SalePrice", data=df)
    plt.title("Average SalePrice by House Age")
    plt.xticks(rotation=45)
    path = "static/eda/saleprice_vs_house_age.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # TotalBath
    plt.figure(figsize=(12,8))
    sns.barplot(x="TotalBath", y="SalePrice", data=df)
    plt.title("Average SalePrice by Total Number of Bath")
    plt.xticks(rotation=45)
    path = "static/eda/saleprice_vs_total_bath.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # Categorical Features
    # Neighborhood
    plt.figure(figsize=(12,8))
    sns.boxplot(x="Neighborhood", y="SalePrice", data=df)
    plt.title("Average SalePrice by Neighborhood")
    plt.xticks(rotation=45)
    path = "static/eda/saleprice_vs_neighborhood.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # ExterQual
    plt.figure(figsize=(10,6))
    sns.boxplot(x="ExterQual", y="SalePrice", data=df)
    plt.title("Average SalePrice by Exterior Quality")
    plt.xticks(rotation=45)
    path = "static/eda/saleprice_vs_exterior_quality.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # BsmtQual
    plt.figure(figsize=(10,6))
    sns.barplot(x="BsmtQual", y="SalePrice", data=df)
    plt.title("Average SalePrice by Basement Quality")
    plt.xticks(rotation=45)
    path = "static/eda/saleprice_vs_basement_quality.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # KitchenQual
    plt.figure(figsize=(10,6))
    sns.boxplot(x="KitchenQual", y="SalePrice", data=df)
    plt.title("Average SalePrice by Kitchen Quality")
    plt.xticks(rotation=45)
    path = "static/eda/saleprice_vs_kitchen_quality.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # SaleCondition
    plt.figure(figsize=(10,6))
    sns.barplot(x="SaleCondition", y="SalePrice", data=df)
    plt.title("Average SalePrice by Sale Condition")
    plt.xticks(rotation=45)
    path = "static/eda/saleprice_vs_sale_condition.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    # HouseStyle
    plt.figure(figsize=(12,8))
    sns.boxplot(x="HouseStyle", y="SalePrice", data=df)
    plt.title("Average SalePrice by House Style")
    plt.xticks(rotation=45)
    path = "static/eda/saleprice_vs_house_style.png"
    plt.savefig(path)
    plots.append(path)
    plt.close()

    return plots