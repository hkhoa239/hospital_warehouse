import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sqlalchemy import create_engine
import shap

# Kết nối đến database trên Neon
@st.cache_resource
def connect_db():
    engine = create_engine("postgresql://hospital_owner:npg_J16pOhzLtlIe@ep-purple-haze-a1ykx9rh-pooler.ap-southeast-1.aws.neon.tech/hospital?sslmode=require")
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
            aq."AQI",
            wc."MAX_TEMP", wc."MIN_TEMP", wc."HUMIDITY",
            fhe.outcome_id
        FROM fact_hospital_event fhe
        LEFT JOIN dim_demographics dm ON fhe.sno = dm.sno
        LEFT JOIN dim_patient_history ph ON fhe.sno = ph.sno
        LEFT JOIN dim_heart_disease hd ON fhe.sno = hd.sno
        LEFT JOIN dim_other_comorbidities od ON fhe.sno = od.sno
        LEFT JOIN dim_lab_test lab ON fhe.sno = lab.sno
        LEFT JOIN dim_air_quality aq ON fhe.admission_date_id = aq.date_id
        LEFT JOIN dim_weather_condition wc ON fhe.admission_date_id = wc.date_id
        WHERE fhe.outcome_id IS NOT NULL
    """
    df = pd.read_sql(query, _engine)
    return df

# Tiền xử lý
def preprocess(df):
    df = df.dropna() 
    df['outcome'] = (df['outcome_id'] == "O2").astype(int)
    df = df.drop(columns=['sno', 'outcome_id'])
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  
    df = pd.get_dummies(df, drop_first=True)
    df = df.fillna(0)

    return df

# Huấn luyện và biểu diễn biểu đồ
def train_and_plot(df):
    X = df.drop(columns='outcome')
    y = df['outcome']

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', base_score=0.5)
    model.fit(X, y)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    top10 = feat_imp.sort_values(by='Importance', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top10, y='Feature', x='Importance', palette='viridis', ax=ax)
    ax.set_title("Top 10 features affecting mortality(XGBoost)")
    st.pyplot(fig)
    X = X.astype('float64')
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    st.subheader("SHAP Summary Plot")
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(fig)
    return model, X, y

# So sánh cohort (STEMI vs Non-STEMI hoặc Shock vs Không Shock)
def compare_cohorts(df):
    st.subheader("Comparison of patient groups")

    stemi_df = df[df['STEMI'] == 1]
    non_stemi_df = df[df['STEMI'] == 0]
    st.write(f"STEMI: {stemi_df.shape[0]} patients")
    st.write(f"Non-STEMI: {non_stemi_df.shape[0]} patients")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='STEMI', hue='outcome', ax=ax)
    ax.set_title("Comparison of outcome rates: STEMI vs Non-STEMI")
    st.pyplot(fig)

    shock_df = df[df['SHOCK'] == 1]
    non_shock_df = df[df['SHOCK'] == 0]
    st.write(f"Shock: {shock_df.shape[0]} patients")
    st.write(f"Non-Shock: {non_shock_df.shape[0]} patients")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x='SHOCK', hue='outcome', ax=ax)
    ax.set_title("Comparison of outcome rates: Shock vs Không Shock")
    st.pyplot(fig)

# UI
def main():
    st.title("Analysis of characteristics influencing mortality")
    st.write("An application using XGBoost to identify factors influencing mortality outcomes from hospital data.")

    engine = connect_db()
    df = load_data(engine)
    if df.empty:
        st.warning("No suitable data available.")
    else:
        df_processed = preprocess(df)
        train_and_plot(df_processed)
        compare_cohorts(df_processed)

if __name__ == "__main__":
    main()
