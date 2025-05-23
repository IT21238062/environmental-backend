from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
from datetime import datetime

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow mobile app to connect

# Global variable to store the loaded model
model = None

def load_model():
    """Load the trained model when server starts"""
    global model
    try:
        print("🛠 Current Directory:", os.getcwd())
        print("📁 Files in directory:", os.listdir())

        model_file = 'my_model.joblib'
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            print("✓ Model loaded successfully from my_model.joblib!")
            return True
        else:
            print("✗ Error: my_model.joblib not found!")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False

# Load model immediately (for both dev server and Gunicorn)
load_model()

@app.route('/')
def home():
    return jsonify({
        "message": "Environmental Control API is working!",
        "status": "success",
        "version": "1.0",
        "system": "HVAC Environmental Monitoring Control",
        "model_status": "loaded" if model is not None else "not_loaded",
        "endpoints": {
            "health": "/",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model_info",
            "system_info": "/health"
        }
    })

@app.route('/health')
def health_check():
    return jsonify({
        "server_status": "running",
        "model_status": "loaded" if model is not None else "not_loaded",
        "model_file": "my_model.joblib",
        "system": "Environmental Control System",
        "timestamp": datetime.now().isoformat(),
        "features": ["airflow", "humidity", "temperature"],
        "prediction_range": "1-5 (airflow levels)"
    })

@app.route('/predict', methods=['POST'])
def predict_airflow_level():
    try:
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model not loaded. Please ensure my_model.joblib is available.",
                "success": False
            }), 500

        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided",
                "success": False
            }), 400

        required_fields = ['airflow', 'humidity', 'temperature']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Missing required field: {field}",
                    "required_fields": required_fields,
                    "success": False
                }), 400

        try:
            airflow = float(data['airflow'])
            humidity = float(data['humidity'])
            temperature = float(data['temperature'])
        except (ValueError, TypeError) as e:
            return jsonify({
                "status": "error",
                "message": f"Invalid data types. All fields must be numbers. Error: {str(e)}",
                "success": False
            }), 400

        validation_warnings = []
        if not (50 <= airflow <= 500):
            validation_warnings.append("Airflow outside recommended range (50-500)")
        if not (0 <= humidity <= 100):
            validation_warnings.append("Humidity outside valid range (0-100%)")
        if not (10 <= temperature <= 40):
            validation_warnings.append("Temperature outside recommended range (10-40°C)")

        input_features = np.array([[airflow, humidity, temperature]])
        prediction = model.predict(input_features)
        predicted_level = max(1, min(5, int(prediction[0])))

        try:
            prediction_proba = model.predict_proba(input_features)
            confidence = float(max(prediction_proba[0])) * 100
        except:
            confidence = 85.0

        response = {
            "status": "success",
            "prediction": {
                "airflow_level": predicted_level,
                "confidence": round(confidence, 1),
                "recommendation": get_level_description(predicted_level)
            },
            "input": {
                "airflow": airflow,
                "humidity": humidity,
                "temperature": temperature
            },
            "timestamp": datetime.now().isoformat(),
            "success": True
        }

        if validation_warnings:
            response["warnings"] = validation_warnings

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}",
            "success": False,
            "timestamp": datetime.now().isoformat()
        }), 500

def get_level_description(level):
    descriptions = {
        1: "Minimal airflow - Good environmental conditions",
        2: "Low airflow - Slightly elevated conditions",
        3: "Medium airflow - Normal operating conditions",
        4: "High airflow - Poor conditions, increased ventilation needed",
        5: "Maximum airflow - Critical conditions, full ventilation required"
    }
    return descriptions.get(level, "Unknown level")

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if model is None:
            return jsonify({"status": "error", "message": "Model not loaded", "success": False}), 500

        data = request.get_json()
        if not data or 'readings' not in data:
            return jsonify({"status": "error", "message": "Please provide 'readings' array", "success": False}), 400

        readings = data['readings']
        if not isinstance(readings, list):
            return jsonify({"status": "error", "message": "Readings must be an array", "success": False}), 400

        results = []
        for i, reading in enumerate(readings):
            try:
                airflow = float(reading['airflow'])
                humidity = float(reading['humidity'])
                temperature = float(reading['temperature'])
                input_features = np.array([[airflow, humidity, temperature]])
                prediction = model.predict(input_features)
                predicted_level = max(1, min(5, int(prediction[0])))

                try:
                    prediction_proba = model.predict_proba(input_features)
                    confidence = float(max(prediction_proba[0])) * 100
                except:
                    confidence = 85.0

                results.append({
                    "index": i,
                    "airflow_level": predicted_level,
                    "confidence": round(confidence, 1),
                    "recommendation": get_level_description(predicted_level),
                    "input": reading
                })

            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e),
                    "input": reading
                })

        return jsonify({
            "status": "success",
            "results": results,
            "total_readings": len(readings),
            "successful_predictions": len([r for r in results if 'error' not in r]),
            "timestamp": datetime.now().isoformat(),
            "success": True
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e), "success": False}), 500

@app.route('/model_info')
def model_info():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded", "success": False}), 500
    try:
        info = {
            "status": "success",
            "success": True,
            "model_type": type(model).__name__,
            "model_file": "my_model.joblib",
            "features": ["airflow", "humidity", "temperature"],
            "target": "airflow_level",
            "possible_predictions": [1, 2, 3, 4, 5],
            "prediction_meanings": {
                "1": "Minimal airflow (good conditions)",
                "2": "Low airflow",
                "3": "Medium airflow (normal)",
                "4": "High airflow (poor conditions)",
                "5": "Maximum airflow (critical)"
            }
        }

        if hasattr(model, 'feature_importances_'):
            features = ["airflow", "humidity", "temperature"]
            info["feature_importance"] = {
                f: round(float(i), 3) for f, i in zip(features, model.feature_importances_)
            }

        return jsonify(info)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to get model info: {str(e)}",
            "success": False
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "success": False,
        "available_endpoints": ["/", "/health", "/predict", "/batch_predict", "/model_info"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "success": False
    }), 500

if __name__ == '__main__':
    print("Starting Environmental Control API Server...")
    print("=" * 50)
    print("Server starting on http://localhost:5000")
    print("Available endpoints:")
    print("  GET  /           - Home page")
    print("  GET  /health     - Health check")
    print("  POST /predict    - Single prediction")
    print("  POST /batch_predict - Multiple predictions")
    print("  GET  /model_info - Model information")
    print("=" * 50)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
