import joblib
import os
import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def save_model(model, metadata, scaler=None):
    """Saves the trained model, its metadata, and optionally a scaler."""
    # Vercel has a read-only filesystem. Prevent crashes on write attempts.
    if os.environ.get('VERCEL'):
        print("[Warning] Attempted to save model in a read-only environment (Vercel). Skipping write.")
        return
        
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    if scaler:
        joblib.dump(scaler, SCALER_PATH)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model and metadata saved to {MODEL_DIR}")

def load_model():
    """Loads the trained model, its metadata, and scaler if available."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None, None, None
    
    model = joblib.load(MODEL_PATH)
    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    return model, metadata, scaler

def get_model_status():
    """Checks if a model has been trained and saved."""
    return os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH)