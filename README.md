# House Price Prediction Web App

A web application to train models and predict the price of houses; based on the Ames Housing Dataset. Users can explore the dataset (EDA), train different machine learning models, and make predictions by manually inserting values.

---

## Features

- **Exploratory Data Analysis (EDA)**: View statistics, distributions and correlation plots of key features and some derived features.
- **Model Training**: Train models by giving them a name and selecting the features that will be used, among those of the dataset; there are several training models available:
    - Linear Regressor
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - Gradient Descent Regressor
- **Model Management**:
    - View all trained models with their characteristics
    - Delete models

---

## Requirements

- Docker
- Docker Compose (latest version)

## Running the App using Docker Compose

1. Build and start the container:

```bash
docker compose up --build
```

2. Open your Browser:

http://localhost:5XXX

The application selects an available port starting from port 5000; you can see which port is being used by running the following command in another terminal:
```bash
docker ps
```

3. Stop the container:

Press Ctrl+C in the terminal or run (in another terminal):
```bash
docker compose down
```
