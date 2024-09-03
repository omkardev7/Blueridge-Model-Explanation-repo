# Forecasting Models Chatbot

This project is a Streamlit web application that applies three different forecasting models (Moving Average, Croston, and Holt-Winters) to predict demand based on historical data. The application also integrates a language model (LLM) from OpenAI to explain the results and answer questions related to the forecasting models.

## Features

- **Data Upload**: Upload a CSV file containing historical demand data with `Date` and `Demand` columns.
- **Forecasting Models**: 
  - Moving Average
  - Croston
  - Holt-Winters Exponential Smoothing
- **Model Performance Comparison**: Calculates and compares the Root Mean Squared Error (RMSE) for each model.
- **LLM Explainer**: Uses OpenAI's GPT-based model to explain the forecasting results and answer user queries.
- **Interactive UI**: User-friendly interface with options to submit additional questions about the models or results.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Streamlit
- NumPy
- pandas
- scikit-learn
- statsmodels
- openai

## Usage

1. **Upload a CSV File**: The CSV file should contain two columns: `Date` (YYYY-MM-DD format) and `Demand`.
2. **View Forecasts and Explanations**: The app will display forecasts from three models, compare their performances, and select the best model.
3. **Ask Additional Questions**: Users can enter additional questions to get more insights or explanations about the forecasts.
