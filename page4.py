import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sqlalchemy import create_engine
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
connection_string = os.getenv('DATABASE_URL')

# Kết nối đến database trên Neon
@st.cache_resource
def connect_db():
    engine = create_engine(connection_string)
    return engine

# Load dữ liệu từ các bảng
@st.cache_data
def load_data(_engine):
    query = """
        SELECT 
            fhe.sno,
            dm.age, dm.gender, dm.rural,
            ph."SMOKING", ph."ALCOHOL", ph."DM", ph."HTN", ph."CAD",
            hd."HEART_FAILURE", hd."STEMI", hd."VALVULAR",
            od."CHEST_INFECTION", od."SHOCK",
            lab."HB", lab."TLC", lab."CREATININE", lab."BNP", lab."EF",
            aq."AQI", aq."PROMINENT_POLLUTENT", aq."PM2.5_AVG", aq."PM10_AVG", 
            aq."NO2_AVG", aq."NH3_AVG", aq."SO2_AVG", aq."CO_AVG", aq."OZONE_AVG",
            wc."MAX_TEMP", wc."MIN_TEMP", wc."HUMIDITY",
            fhe.outcome_id, ad.date AS admission_date, dis.date AS discharge_date
        FROM fact_hospital_event fhe
        LEFT JOIN dim_demographics dm ON fhe.sno = dm.sno
        LEFT JOIN dim_patient_history ph ON fhe.sno = ph.sno
        LEFT JOIN dim_heart_disease hd ON fhe.sno = hd.sno
        LEFT JOIN dim_other_comorbidities od ON fhe.sno = od.sno
        LEFT JOIN dim_lab_test lab ON fhe.sno = lab.sno
        LEFT JOIN dim_air_quality aq ON fhe.admission_date_id = aq.date_id
        LEFT JOIN dim_weather_condition wc ON fhe.admission_date_id = wc.date_id
        LEFT JOIN dim_admission_date ad ON fhe.admission_date_id = ad.date_id
        LEFT JOIN dim_discharge_date dis ON fhe.discharge_date_id = dis.date_id
        WHERE fhe.outcome_id IS NOT NULL
    """
    df = pd.read_sql(query, _engine)
    return df

# Tiền xử lý
def preprocess(df):
    df = df.dropna()
    df['outcome'] = (df['outcome_id'] == "O2").astype(int)  
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df['day'] = df['admission_date'].dt.day
    df['month'] = df['admission_date'].dt.month
    df['year'] = df['admission_date'].dt.year
    return df

# Hiển thị biểu đồ Line: AQI và số ca tử vong theo thời gian
def plot_aqi_vs_mortality(df):
    grouped = df.groupby(['year', 'month']).agg({'AQI': 'mean', 'outcome': 'sum'}).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(grouped['month'], grouped['AQI'], color='tab:blue', label='AQI', marker='o')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average AQI', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(grouped['month'], grouped['outcome'], color='tab:red', label='Deaths', marker='o')
    ax2.set_ylabel('Number of Deaths', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    st.title('AQI and Number of Deaths Over Time')
    st.write(grouped)
    st.pyplot(fig)

# Hiển thị biểu đồ Line: Nhiệt độ, độ ẩm và số ca nhập viện
def plot_co_avg_and_admissions(df):
    grouped = df.groupby(['year', 'month']).agg({'CO_AVG': 'mean', 'HUMIDITY': 'mean', 'sno': 'count'}).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(grouped['month'], grouped['CO_AVG'], label='CO avg', color='tab:orange', marker='o')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('CO Averag', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')
    
    ax2 = ax1.twinx()
    ax2.plot(grouped['month'], grouped['sno'], label='Admissions', color='tab:purple', marker='o')
    ax2.set_ylabel('Number of Admissions', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    
    st.title('Weather Temperature and Number of Admissions Over Time')
    st.write(grouped)
    st.pyplot(fig)

def plot_nh3_avg_and_admissions(df):
    grouped = df.groupby(['year', 'month']).agg({'NH3_AVG': 'mean', 'HUMIDITY': 'mean', 'sno': 'count'}).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(grouped['month'], grouped['NH3_AVG'], label='CO avg', color='tab:orange', marker='o')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('NH3 Averag', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')
    
    ax2 = ax1.twinx()
    ax2.plot(grouped['month'], grouped['sno'], label='Admissions', color='tab:purple', marker='o')
    ax2.set_ylabel('Number of Admissions', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    
    st.title('NH3 Average and Number of Admissions Over Time')
    st.write(grouped)
    st.pyplot(fig)

def plot_pm10_avg_and_admissions(df):
    grouped = df.groupby(['year', 'month']).agg({'PM10_AVG': 'mean', 'HUMIDITY': 'mean', 'sno': 'count'}).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(grouped['month'], grouped['PM10_AVG'], label='CO avg', color='tab:orange', marker='o')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('PM10 Averag', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')
    
    ax2 = ax1.twinx()
    ax2.plot(grouped['month'], grouped['sno'], label='Admissions', color='tab:purple', marker='o')
    ax2.set_ylabel('Number of Admissions', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    
    st.title('PM10 Average and Number of Admissions Over Time')
    st.write(grouped)
    st.pyplot(fig)

def plot_min_temp_and_deaths(df):
    grouped = df.groupby(['year', 'month']).agg({'MAX_TEMP': 'mean', 'MIN_TEMP': 'mean', 'HUMIDITY': 'mean', 'outcome': 'sum'}).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(grouped['month'], grouped['MIN_TEMP'], label='Min Temp', color='tab:orange', marker='o')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Min Temp', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')
    
    ax2 = ax1.twinx()
    ax2.plot(grouped['month'], grouped['outcome'], label='Deaths', color='tab:purple', marker='o')
    ax2.set_ylabel('Number of Deaths', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    
    st.title('Minimum Temperature and Number of Deaths Over Time')
    st.write(grouped)
    st.pyplot(fig)

def plot_no2_and_deaths(df):
    grouped = df.groupby(['year', 'month']).agg({'NO2_AVG': 'mean', 'outcome': 'sum'}).reset_index()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(grouped['month'], grouped['NO2_AVG'], label='Min Temp', color='tab:orange', marker='o')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('NO2 average', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')
    
    ax2 = ax1.twinx()
    ax2.plot(grouped['month'], grouped['outcome'], label='Deaths', color='tab:purple', marker='o')
    ax2.set_ylabel('Number of Deaths', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    
    st.title('NO2 Average and Number of Deaths Over Time')
    st.write(grouped)
    st.pyplot(fig)

# Boxplot: AQI theo outcome
def plot_aqi_boxplot(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='outcome', y='AQI', data=df, ax=ax)
    ax.set_xlabel('Outcome')
    ax.set_ylabel('AQI')
    st.title('AQI by Outcome (Discharged vs Expired)')
    st.pyplot(fig)

# Heatmap tương quan giữa các yếu tố môi trường và lab test
def plot_correlation_heatmap_admiss(df):
    grouped = df.groupby(['year', 'month']).agg({'sno': 'count','AQI': 'mean', 'MAX_TEMP': 'mean', 'MIN_TEMP': 'mean','HUMIDITY': 'mean','PM2.5_AVG': 'mean','PM10_AVG': 'mean','NO2_AVG': 'mean','NH3_AVG': 'mean','SO2_AVG': 'mean','CO_AVG': 'mean','OZONE_AVG': 'mean'}).reset_index()
    corr_df = grouped[['sno','AQI', 'MAX_TEMP', 'MIN_TEMP','HUMIDITY','PM2.5_AVG','PM10_AVG','NO2_AVG','NH3_AVG','SO2_AVG','CO_AVG','OZONE_AVG']].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
    st.title('Correlation Heatmap: Environmental Factors and Admissions')
    st.pyplot(fig)

def plot_correlation_heatmap_death(df):
    grouped = df.groupby(['year', 'month']).agg({'outcome': 'sum','AQI': 'mean', 'MAX_TEMP': 'mean', 'MIN_TEMP': 'mean','HUMIDITY': 'mean','PM2.5_AVG': 'mean','PM10_AVG': 'mean','NO2_AVG': 'mean','NH3_AVG': 'mean','SO2_AVG': 'mean','CO_AVG': 'mean','OZONE_AVG': 'mean'}).reset_index()
    corr_df = grouped[['outcome','AQI', 'MAX_TEMP', 'MIN_TEMP','HUMIDITY','PM2.5_AVG','PM10_AVG','NO2_AVG','NH3_AVG','SO2_AVG','CO_AVG','OZONE_AVG']].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
    st.title('Correlation Heatmap: Environmental Factors and Number of deaths')
    st.pyplot(fig)

# Hàm main
def main():
    st.title("Hospital Data Analysis and Mortality Prediction")

    # Kết nối DB và load dữ liệu
    engine = connect_db()
    df = load_data(engine)
    df_processed = preprocess(df)
    
    # Dropdown lọc theo tháng/năm
    year = st.selectbox('Select Year', df_processed['year'].unique())
    filtered_df = df_processed[(df_processed['year'] == year)]

    plot_correlation_heatmap_admiss(filtered_df)
    plot_correlation_heatmap_death(filtered_df)

    if year == 2017:
        plot_co_avg_and_admissions(filtered_df)
        plot_aqi_vs_mortality(filtered_df)
        plot_min_temp_and_deaths(filtered_df)
        
    if year == 2018:
        plot_nh3_avg_and_admissions(filtered_df)
        plot_no2_and_deaths(filtered_df)
        plot_min_temp_and_deaths(filtered_df)
    if year == 2019:
        plot_pm10_avg_and_admissions(filtered_df)
        plot_no2_and_deaths(filtered_df)
        plot_min_temp_and_deaths(filtered_df)


if __name__ == "__main__":
    main()
