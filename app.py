import streamlit as st
import numpy as np
import os
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from openai import OpenAI
os.environ["OPENAI_API_KEY"] =OPENAI_API_KEY
# # Load data
# def load_data(file):
#     df = pd.read_csv(file, index_col=False)
#     df = df[['Date', 'Demand']]
#     return df
# Load data
def load_data(file):
    try:
        df = pd.read_csv(file, index_col=False)
        if df.empty or 'Date' not in df.columns or 'Demand' not in df.columns:
            st.error("Uploaded file is empty or does not contain the required columns: 'Date' and 'Demand'.")
            return None
        df = df[['Date', 'Demand']]
        return df
    except pd.errors.EmptyDataError:
        st.error("Uploaded file is empty.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# Moving Average Calculation
def moving_average(ts, window_size):
    return ts.rolling(window=window_size).mean()

# Croston Model
def Croston(ts, extra_periods=6, alpha=0.4):
    d = np.array(ts)
    cols = len(d)
    d = np.append(d, [np.nan] * extra_periods)
    
    a, p, f = np.full((3, cols + extra_periods), np.nan)
    q = 1
    
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0] / p[0]
    
    for t in range(0, cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = alpha * q + (1 - alpha) * p[t]
            f[t + 1] = a[t + 1] / p[t + 1]
            q = 1
        else:
            a[t + 1] = a[t]
            p[t + 1] = p[t]
            f[t + 1] = f[t]
            q += 1
    
    a[cols + 1:cols + extra_periods] = a[cols]
    p[cols + 1:cols + extra_periods] = p[cols]
    f[cols + 1:cols + extra_periods] = f[cols]
    
    df = pd.DataFrame({
        "Demand": d,
        "Forecast": f,
        "Period": p,
        "Level": a,
        "Error": d - f
    })
    return df

# Holt-Winters Model
def fit_holt_winters(train):
    model = ExponentialSmoothing(train['Demand'], trend='add', seasonal='add', seasonal_periods=12)
    return model.fit()

def calculate_rmse(actual, forecast):
    return np.sqrt(mean_squared_error(actual, forecast))

# LLM Functions
def create_prompt(rmse_values, comparison_df):
    best_model = min(rmse_values, key=rmse_values.get)
    
    prompt = f"""
    We have applied three different forecasting models to predict demand: Moving Average, Croston, and Holt-Winters.
    Here are the RMSE values for each model: {rmse_values}.
    
    The best-performing model is {best_model} as it has the lowest RMSE value.
    
    The following pandas dataframe shows the actual demand and the forecasted values by each model Date format(YYYY-MM-DD):

    {comparison_df}   
 
    Please answer the following questions based on this information (Note:generate answers based on rmse_values & comparison_df uploaded only and if not found then say can't access rmse_values & comparison_df):

    Type 1 Questions:
    1. What is the prediction for September 2024? (Use the forecasted value for September from the best model)
    2. What is the prediction for the next six months? (Provide the next six months' forecasted values from the best model)

    Type 2 Questions:
    1. What model have you used for this prediction?(Also mention out off other models)
    2. What are the parameters of the model?
    3. Why was this model used?

    Type 3 Questions:
    1. Tell me more about how the moving average works.
    2. Explain to me what Holt-Winters does.
    

    """

    return prompt

def create_prompt_additional(rmse_values, comparison_df, additional_question=None):
    best_model = min(rmse_values, key=rmse_values.get)
    
    prompt = f"""
    We have applied three different forecasting models to predict demand: Moving Average, Croston, and Holt-Winters.
    Here are the RMSE values for each model: {rmse_values}.
    
    The best-performing model is {best_model} as it has the lowest RMSE value.
    
    The following pandas dataframe shows the actual demand and the forecasted values by each model Date format(YYYY-MM-DD):
    {comparison_df}
   
    Please answer the following question based on this information:
    Additional Question:
    {additional_question}
    """
    
    return prompt

def get_completion(prompt):
    openai_client = OpenAI(api_key=OPENAI_API_KEY)  # Replace with your OpenAI API key
    
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {"role": "system", 
            "content": "You are an AI assistant providing explanations for different forecasting models and their results."
            },
            {"role": "user", 
            "content": prompt}
        ],
        model="gpt-4o-mini",  # Use 'gpt-4-mini' or the appropriate model if needed
        temperature=0.3,
    )
    output = chat_completion.choices[0].message.content
    return output

def llm_explainer(rmse_values, comparison_df):
    prompt = create_prompt(rmse_values, comparison_df)
    response = get_completion(prompt)
    return response

def llm_explainer_additional(rmse_values, comparison_df, additional_question):
    prompt = create_prompt_additional(rmse_values, comparison_df, additional_question)
    response = get_completion(prompt)
    return response

# Streamlit UI
st.title("Forecasting Models Chatbot")

# Upload CSV File
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("Data Loaded Successfully.")
    
    # Split data
    split_index = int(len(df) * 0.8)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    
    # Moving Average Forecast
    window_size = train.count()['Demand']
    train['MA_Forecast'] = moving_average(train['Demand'], window_size)
    ma_forecast = moving_average(df['Demand'], window_size).dropna()
    ma_forecast_test = ma_forecast[-len(test):]
    ma_rmse = calculate_rmse(test['Demand'], ma_forecast_test)
    
    # Croston Model Forecast
    croston_results = Croston(df['Demand'], extra_periods=6, alpha=0.5)
    croston_forecast = croston_results['Forecast'][-len(test):].values
    croston_rmse = calculate_rmse(test['Demand'], croston_forecast)
    
    # Holt-Winters Forecast
    holt_winters_model = fit_holt_winters(train)
    holt_winters_forecast = holt_winters_model.forecast(steps=len(test))
    holt_winters_rmse = calculate_rmse(test['Demand'], holt_winters_forecast)
     
    # Determine Best Model
    rmse_values = {
        'Moving Average': ma_rmse,
        'Croston': croston_rmse,
        'Holt-Winters': holt_winters_rmse
    }
    best_model = min(rmse_values, key=rmse_values.get)
    
    # Comparison DataFrame
    comparison_df = pd.DataFrame({
        'Date': test['Date'],
        'Actual Demand': test['Demand'],
        'Moving Average Forecast': ma_forecast_test.values,
        'Croston Forecast': croston_forecast,
        'Holt-Winters Forecast': holt_winters_forecast.values
    }, index=test.index)
    
    
    # LLM Explainer Response with Spinner
    with st.spinner('Generating LLM Explanation...'):
        initial_response = llm_explainer(rmse_values, comparison_df)       
    # Initial LLM Explanation
    # st.header("LLM Explainer Response")
    # st.write(initial_response)
        st.markdown(
        "<h1 style='color: red; font-weight: bold; font-size: 36px;'>LLM Explainer Response</h1>", 
        unsafe_allow_html=True
    )
    st.write(initial_response)
    
    st.markdown(
        "<h1 style='color: red; font-weight: bold; font-size: 36px;'>Details</h1>", 
        unsafe_allow_html=True
    )    
    st.subheader("Forecast Comparison")
    st.dataframe(comparison_df)
    # print(comparison_df)
    
    # Display RMSE Values
    st.subheader("Model Performance")
    st.write(f'Moving Average RMSE: {ma_rmse}')
    st.write(f'Croston RMSE: {croston_rmse}')
    st.write(f'Holt-Winters RMSE: {holt_winters_rmse}')
    st.write(f'The best performing model is: {best_model}')   
    
    # Additional Query Input
    st.markdown(
        "<h1 style='color: red; font-weight: bold; font-size: 36px;'>Do you have any additional questions?</h1>", 
        unsafe_allow_html=True
    )   
    additional_query = st.text_input("Enter Your Question:")
    
    if st.button("Submit Additional Question"):
        if additional_query:
            with st.spinner('Generating Response to Additional Question...'):
                additional_response = llm_explainer_additional(rmse_values, comparison_df, additional_query)
            st.write("LLM Response to Additional Question:")
            st.write(additional_response)
    

