import gradio as gr
import pandas as pd
import numpy as np
import os
from ml_pipeline import MLPipeline
from model_manager import get_model_status, load_model

# Initialize Pipeline
pipeline = MLPipeline()

def get_status():
    is_trained = get_model_status()
    if is_trained:
        return "✅ Model Status: Trained & Ready"
    return "⚠️ Model Status: Not Found (Please Train first)"

def train_model(target_col):
    if not pipeline.load_data():
        return "Error: Dataset not found in 'dataset/' folder."
    try:
        results = pipeline.train(target_col)
        return f"Success! Accuracy: {results['accuracy']:.2%} | Precision: {results['precision']:.2%}"
    except Exception as e:
        return f"Error: {str(e)}"

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    try:
        result = pipeline.predict(input_dict)
        return f"Prediction: {result['prediction_label']}\nConfidence: {result['probability']:.1%}"
    except Exception as e:
        return f"Error: {str(e)}"

# Custom Theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill='*neutral_950',
    block_background_fill='*neutral_900',
    block_label_text_color='*neutral_200',
)

with gr.Blocks(theme=theme, title="NEXUS ML Diagnostic") as demo:
    gr.Markdown("# 🛡️ NEXUS ML Diagnostic Center")
    gr.Markdown("A stable alternative for heart disease screening and model management.")
    
    with gr.Tab("📊 Analytics & Training"):
        status_label = gr.Markdown(get_status())
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model Synthesis")
                if pipeline.load_data():
                    target_dropdown = gr.Dropdown(choices=list(pipeline.df.columns), label="Target Column", value="num")
                    train_btn = gr.Button("🚀 Train Naive Bayes Model", variant="primary")
                    train_out = gr.Textbox(label="Training Result")
                    train_btn.click(train_model, inputs=[target_dropdown], outputs=[train_out])
                else:
                    gr.Error("Dataset not found.")

    with gr.Tab("🔮 Prediction Engine"):
        gr.Markdown("### Patient Data Entry")
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Age", value=50)
                sex = gr.Slider(0, 1, step=1, label="Sex (0=F, 1=M)", value=1)
                cp = gr.Slider(0, 3, step=1, label="Chest Pain Type (0-3)", value=2)
                trestbps = gr.Number(label="Resting BP", value=130)
                chol = gr.Number(label="Cholesterol", value=240)
            with gr.Column():
                fbs = gr.Slider(0, 1, step=1, label="Fasting Blood Sugar > 120", value=0)
                restecg = gr.Slider(0, 2, step=1, label="Resting ECG", value=1)
                thalach = gr.Number(label="Max Heart Rate", value=150)
                exang = gr.Slider(0, 1, step=1, label="Exercise Angina", value=0)
            with gr.Column():
                oldpeak = gr.Number(label="ST Depression", value=1.0)
                slope = gr.Slider(0, 2, step=1, label="ST Slope", value=1)
                ca = gr.Slider(0, 3, step=1, label="Major Vessels", value=0)
                thal = gr.Slider(0, 3, step=1, label="Thalassemia", value=2)
        
        predict_btn = gr.Button("🔍 Perform Diagnostic", variant="primary")
        predict_out = gr.Textbox(label="Diagnostic Outcome", interactive=False)
        
        predict_btn.click(
            predict_heart_disease,
            inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
            outputs=[predict_out]
        )

if __name__ == "__main__":
    demo.launch(server_port=7860)
