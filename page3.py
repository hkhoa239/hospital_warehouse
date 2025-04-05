import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sqlalchemy import create_engine
import streamlit.components.v1 as components

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
    df = pd.get_dummies(df, drop_first=False)
    return df

# Lọc bệnh nhân theo mrd_no
def filter_patient(df, mrd_no):
    return df[df['sno'] == mrd_no].iloc[0]

# Huấn luyện mô hình XGBoost
def train_model(df):
    X = df.drop(columns='outcome', axis=1)
    y = df['outcome']
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', base_score=0.5)
    model.fit(X, y)
    return model, X, y

# Hiển thị hồ sơ bệnh nhân
def display_patient_record(patient):
    st.write(f"### Patient profile:")
    st.write(f"- Age: {patient['age']}")
    if patient['gender']=="M":
        st.write(f"- Gender: Male")
    else:
        st.write(f"- Gender: Female")
    st.write(f"- Medical condition:")
    fields = ['SMOKING', 'ALCOHOL', 'DM', 'HTN', 'CAD', 'HEART_FAILURE', 
          'STEMI', 'VALVULAR', 'CHEST_INFECTION', 'SHOCK']
    data = []
    for i in fields:
        tmp = 'YES' if patient[i] else 'NO'
        data.append({'Condition': i, 'Status': tmp})
    table = pd.DataFrame(data)
    st.table(table)
    st.write(f"- Lab test result:")
    lab_fields = ['HB', 'TLC', 'CREATININE', 'BNP', 'EF']
    lab_data = [{'Lab Test': i, 'Value': patient[i]} for i in lab_fields]
    lab_table = pd.DataFrame(lab_data)
    st.table(lab_table)

# Hiển thị SHAP Force Plot
def shap_force_plot(model, X, patient_row):
    X = X.astype('float64')
    explainer = shap.Explainer(model, X)
    shap_values = explainer(patient_row)
    if shap_values is None or len(shap_values.values) == 0:
        st.write("There is no SHAP value to display!")
        return
    shap.initjs()
    st.subheader("SHAP Force Plot")
    st.pyplot(shap.force_plot(
        base_value=shap_values.base_values[0],
        shap_values=shap_values.values[0],
        features=patient_row,
        show=False,
        matplotlib=True 
    ))

# Giải thích nguyên nhân tử vong và khuyến nghị điều trị
def treatment_recommendations(patient):
    recommendations = []
    if patient['SHOCK'] == 1:
        recommendations.append("The patient shows signs of shock and requires urgent intervention!")
    if patient['STEMI'] == 1:
        recommendations.append("The patient shows signs of STEMI and requires immediate vascular intervention!")
    if recommendations == []:
        recommendations.append("The patient does not require intervention at this time.")
    return recommendations


def main():
    st.title("🔍 Analyzing patients and predicting mortality outcomes")
    
    engine = connect_db()
    df = load_data(engine)

    patient_ids = df['sno'].unique()
    mrd_no = st.selectbox("Select a patient for analysis", patient_ids)
    
    if mrd_no:
        patient = filter_patient(df, mrd_no)
        display_patient_record(patient)
        df_processed = preprocess(df)
        model, X, y = train_model(df_processed)
        patient_index = df[df['sno'] == mrd_no].index[0]
        patient_processed = X.iloc[[patient_index]]
        st.subheader("SHAP Force Plot for the patient")
        shap_force_plot(model, X, patient_processed)
        st.subheader("Recommended treatment:")
        recommendations = treatment_recommendations(patient)
        for rec in recommendations:
            st.write(f"- {rec}")

if __name__ == "__main__":
    main()
