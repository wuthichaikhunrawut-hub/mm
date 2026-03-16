from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import glob
from ml_pipeline import MLPipeline
from model_manager import get_model_status, load_model
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'dataset'
ALLOWED_EXTENSIONS = {'csv', 'arff'}

# Configure Flask App to serve frontend files from the correct directory
app = Flask(__name__,
            template_folder='.',
            static_folder='.',
            static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Checks if the file's extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- HTML Serving Routes ---

@app.route('/')
def home():
    """Premium Home page."""
    return render_template('home.html')

@app.route('/dashboard')
def index():
    """Main dashboard page."""
    # Auto-train if no model exists
    if not get_model_status():
        print("[System] No model found. Initiating auto-synthesis...")
        try:
            pipeline = MLPipeline()
            if pipeline.load_data():
                # Default to 'num' for Heart Disease if present, or first categorical
                target = 'num' if 'num' in pipeline.df.columns else pipeline.df.columns[-1]
                pipeline.train(target)
                print(f"[System] Auto-synthesis complete using target '{target}'.")
        except Exception as e:
            print(f"[System] Auto-synthesis failed: {e}")
            
    return render_template('index.html')

@app.route('/train')
def train_page():
    """Training page."""
    return render_template('train.html')

@app.route('/predict')
def predict_page():
    """Prediction page."""
    return render_template('predict.html')

# --- API Routes ---

@app.route('/api/train', methods=['POST'])
def train_route():
    """Trains the Naive Bayes model."""
    data = request.get_json()
    target_column = data.get('target_column', 'num')
    
    try:
        pipeline = MLPipeline()
        if not pipeline.load_data():
            return jsonify({"error": "Dataset not found"}), 404
            
        results = pipeline.train(target_column)
        return jsonify({"message": "Model trained successfully", "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_route():
    """Predicts the outcome using Naive Bayes."""
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        pipeline_instance = MLPipeline()
        # Extract features if the frontend sends it as {features: {...}}
        features = input_data.get('features', input_data)
        result = pipeline_instance.predict(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dataset/info', methods=['GET'])
def dataset_info_route():
    """Provides column information for form generation."""
    pipeline = MLPipeline()
    if not pipeline.load_data():
        return jsonify({"error": "Dataset not found."}), 404
    
    info = pipeline.get_dataset_info()
    # Also include target_column from model metadata if trained
    is_trained = get_model_status()
    if is_trained:
        _, metadata, _ = load_model()
        info['target_column'] = metadata.get('target_column')
        
    return jsonify(info)

if __name__ == '__main__':
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created '{dataset_dir}' folder. Please add a 'data.csv' file inside it.")
    
    app.run(debug=True, port=5000)