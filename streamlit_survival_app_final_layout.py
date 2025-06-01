
# streamlit_survival_app_final_layout.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# 모델 및 스케일러 로딩
model_os = joblib.load("cox_model_os.pkl")
model_surg = joblib.load("cox_model_surg.pkl")
# 수술용 스케일러
scaler_surg = StandardScaler()
scaler_surg.mean_ = np.array([63.99232246  ,1.0134357  , 1.54403209])
scaler_surg.scale_ = np.array([10.08580931  ,0.77991332 , 0.48392628])

# 생존용 스케일러
scaler_surv = StandardScaler()
scaler_surv.mean_ = np.array([1.55442944 ,2.26919498, 0.42199001 ,1.6604481 ])
scaler_surv.scale_ = np.array([0.12164714 ,1.57492797, 0.15935475, 1.90944101])


st.set_page_config(layout="wide")
st.title("수술 가능성 및 생존 예측")

# ----------------------
# 사이드바: 변수 입력 (고정)
# ----------------------
with st.sidebar:
    #st.header("🛠️ 수술 관련 변수")
    age = st.number_input("Age", value=65)
    anc = st.number_input("ANC (10^3/μL)", value=3.0)
    lymph = st.number_input("Lymphocyte (10^3/μL)", value=1.5)
    meta_count = int(st.selectbox("Metastasis Count", ["0", "1", "2", "3", "4"], index=1))

    #st.header("🌱 생존 관련 변수")
    alb = st.number_input("Albumin (g/dL)", value=4.0)
    mono = st.number_input("Monocyte (10^3/μL)", value=0.4)
    cea = st.number_input("CEA (ng/mL)", value=20.0)
    #alp = st.number_input("ALP (IU/L)", value=100.0)

# ----------------------
# 내부 계산
# ----------------------
log_cea = np.log1p(cea)
log_alb = np.log1p(alb)
log_mono=np.log1p(mono)
#log_alp = np.log1p(alp)
nlr = (anc) / (lymph + 1e-6)
log_nlr = np.log1p(nlr)

df_surg = pd.DataFrame([{
    "AGE": age,
    "META_COUNT": meta_count,
    "LOG_NLR": log_nlr
}])
df_surg_scaled = pd.DataFrame(scaler_surg.transform(df_surg), columns=df_surg.columns)
surg_risk = model_surg.predict_partial_hazard(df_surg_scaled).values[0]

df_surv = pd.DataFrame([{
    "lasso_risk": surg_risk,
    "LOG_ALB": log_alb,
    "LOG_MONO": log_mono,
    "LOG_CEA": log_cea,
}])
df_surv_scaled = pd.DataFrame(scaler_surv.transform(df_surv), columns=df_surv.columns)

# ----------------------
# 예측
# ----------------------
surv = model_os.predict_survival_function(df_surv_scaled)
surg = model_surg.predict_survival_function(df_surg_scaled)

def get_nearest_time_index(df, target_time):
    return df.index.get_indexer([target_time], method='nearest')[0]

times = [365, 1095]
surv_probs = [round(surv.iloc[get_nearest_time_index(surv, t)].values[0], 3) for t in times]
surg_probs = [round(1 - surg.iloc[get_nearest_time_index(surg, t)].values[0], 3) for t in times]

# ----------------------
# 본문: 출력 섹션
# ----------------------
#st.markdown("###  수술 가능 점수")
st.metric("⚠️ 수술 가능 점수 (relative score)", round(surg_risk, 3))

#st.markdown("### 📊 예측 확률 (1년 / 3년)")
result_df = pd.DataFrame({
    "구분": ["생존 확률", "수술 가능성"],
    "1년": [surv_probs[0], surg_probs[0]],
    "3년": [surv_probs[1], surg_probs[1]]
})
st.table(result_df.set_index("구분"))

# ----------------------
# 그래프 섹션
# ----------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📈 수술 가능성 곡선")
    fig1, ax1 = plt.subplots()
    (1 - surg).plot(ax=ax1)
    ax1.axvline(x=365, color='gray', linestyle='--', label='1년')
    ax1.axvline(x=1095, color='gray', linestyle='--', label='3년')
    ax1.set_xlim(0,1800)
    ax1.set_title("Surgery Probability Curve")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Probability")
    st.pyplot(fig1)

with col_right:
    st.markdown("### 📈 생존 곡선")
    fig2, ax2 = plt.subplots()
    surv.plot(ax=ax2)
    ax2.axvline(x=365, color='gray', linestyle='--', label='1년')
    ax2.axvline(x=1095, color='gray', linestyle='--', label='3년')
    ax2.set_xlim(0,1800)
    ax2.set_title("Overall Survival Curve")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Survival Probability")
    st.pyplot(fig2)

