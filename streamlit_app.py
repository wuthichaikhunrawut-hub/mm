import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from ml_pipeline import MLPipeline
from model_manager import get_model_status, load_model

# Page Config
st.set_page_config(
    page_title="NEXUS ML - Professional",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark & Premium)
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#0f172a, #1e293b);
        color: white;
    }
    .main {
        background-color: #0f172a;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
    }
    .stButton>button {
        background-image: linear-gradient(to right, #2563eb, #3b82f6);
        color: white;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    /* Balanced Input Grid */
    [data-testid="column"] {
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Field Mapping for User Friendly Labels
LABEL_MAP = {
    "age": "Age (Years)",
    "sex": "Sex (1=M, 0=F)",
    "cp": "Chest Pain Type",
    "trestbps": "Resting BP (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "fbs": "Fasting Sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Major Vessels (0-3)",
    "thal": "Thalassemia",
    "num": "Diagnosis (Target)"
}

# Initialize Pipeline
@st.cache_resource
def get_pipeline():
    return MLPipeline()

pipeline = get_pipeline()

# Sidebar Info
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=100)
    st.title("🛡️ NEXUS ML")
    st.caption("AI-Powered Diagnostic Workspace")
    st.markdown("---")
    
    st.info("💡 **Tip:** Always clean your data before training for the best accuracy.")
    
    is_trained = get_model_status()
    if is_trained:
        st.success("🤖 **Status**: Model Optimized")
    else:
        st.warning("🤖 **Status**: Analysis Pending")

# Navigation using Tabs (Modern Look)
tab1, tab2, tab3 = st.tabs(["📊 Insights Dashboard", "🏗️ Model Training", "🔮 Prediction Engine"])

with tab1:
    st.title("📊 Data Insights")
    
    if pipeline.load_data():
        info = pipeline.get_dataset_info()
        
        # Stats Overview
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dataset Size", f"{info['rowCount']} Rows")
        m2.metric("Missing Data", f"{info['totalMissing']} Points")
        m3.metric("Features", info['numAttributes'])
        m4.metric("Categories", info['catAttributes'])
        
        st.markdown("---")
        
        # Split Layout
        col_l, col_r = st.columns([2, 1])
        
        with col_l:
            st.subheader("📋 Dataset Preview")
            st.dataframe(pd.DataFrame(info['preview']), use_container_width=True)
            
            st.subheader("📈 Distribution Analysis")
            feature_list = [f for f in info['distributions'].keys()]
            selected_f = st.selectbox("Select Feature to Visualize", feature_list)
            if selected_f:
                dist = info['distributions'][selected_f]
                hist_df = pd.DataFrame({'Value': dist['bins'][:-1], 'Count': dist['counts']})
                st.bar_chart(hist_df.set_index('Value'), color="#3b82f6")
                
        with col_r:
            st.subheader("🔥 Correlations")
            if info['correlationMatrix']:
                try:
                    corr_df = pd.DataFrame(
                        info['correlationMatrix']['values'],
                        index=info['correlationMatrix']['columns'],
                        columns=info['correlationMatrix']['columns']
                    )
                    # Use a standard heatmap without risky stylers if matplotlib is tricky
                    st.write("Heatmap Matrix:")
                    st.dataframe(corr_df.style.background_gradient(axis=None, low=0.75, high=1.0, cmap='Blues'), use_container_width=True)
                except Exception as e:
                    st.error("Correlation view unavailable. Ensure matplotlib is active.")
            else:
                st.info("Limited numeric data.")
    else:
        st.error("No dataset detected. Please ensure data is in the 'dataset/' folder.")

with tab2:
    st.title("🏗️ Model Synthesis")
    
    if pipeline.load_data():
        st.write("Working with:", f"`{pipeline.file_name}`")
        
        target = st.selectbox("Define Target Column", pipeline.df.columns, index=len(pipeline.df.columns)-1)
        
        if st.button("✨ Execute AI Synthesis"):
            with st.status("Synthesizing model...", expanded=True) as s:
                st.write("Extracting features...")
                time.sleep(1)
                st.write("Optimizing Gaussian Naive Bayes parameters...")
                res = pipeline.train(target)
                s.update(label="Synthesis Complete!", state="complete")
            
            st.balloons()
            st.toast("Model successfully trained!")
            
            # Display Metrics Professionaly
            st.markdown("### 🏆 Performance Metrics")
            met1, met2, met3 = st.columns(3)
            met1.metric("Accuracy Score", f"{res['accuracy']:.2%}")
            met2.metric("Precision", f"{res['precision']:.2%}")
            met3.metric("F1-Score", f"{res['f1_score']:.2%}")
            
            with st.expander("Show Detailed Evaluation"):
                st.write("Confusion Matrix:")
                st.table(pd.DataFrame(res['confusion_matrix']))

with tab3:
    st.title("🔮 Prediction Engine")
    
    if is_trained:
        _, metadata, _ = load_model()
        
        st.write(f"Inference Mode: **{metadata['target_column']} Prediction**")
        st.markdown("---")
        
        # Grid Layout for 13 fields (rebalanced into 4 columns)
        input_data = {}
        rows = [metadata['feature_columns'][i:i+4] for i in range(0, len(metadata['feature_columns']), 4)]
        
        for row in rows:
            cols = st.columns(4)
            for i, feature in enumerate(row):
                label = LABEL_MAP.get(feature, feature)
                input_data[feature] = cols[i].number_input(label, value=0.0)
        
        st.markdown("---")
        if st.button("🔍 Perform Diagnostic"):
            try:
                result = pipeline.predict(input_data)
                
                # Full width result card
                res_box = st.container(border=True)
                with res_box:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Predicted Result")
                        st.title(f"Target: {result['prediction_label']}")
                    with c2:
                        st.subheader("AI Confidence")
                        st.metric("Probability", f"{result['probability']:.1%}", delta=result['confidence'])
                        st.progress(result['probability'])
                
            except Exception as e:
                st.error(f"Inference Error: {e}")
    else:
        st.warning("The Prediction Engine requires an optimized model. Please run 'Model Training' first.")
