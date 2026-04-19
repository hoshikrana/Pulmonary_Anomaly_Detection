"""
app/api/routes.py
─────────────────
URL endpoints only. Routes validate → call service → return response.
No business logic, no model code, no image processing.
"""

from flask import Blueprint, request, jsonify, render_template

from app.api.validators import validate_upload, ValidationError
from app.services       import get_inference_service

api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@api_blueprint.route("/result", methods=["GET"])
def result_page():
    return render_template("result.html")


@api_blueprint.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict  —  multipart file upload
    Returns JSON: {status, label, score, raw_mse, confidence,
                   heatmap_b64, reconstruction_b64, original_b64}
    """
    try:
        image_bytes = validate_upload(request)
    except ValidationError as e:
        return jsonify({"error": e.message}), e.status_code

    try:
        result = get_inference_service().predict(image_bytes)
        return jsonify({"status": "success", **result}), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@api_blueprint.route("/api/health", methods=["GET"])
def health():
    """GET /api/health — liveness check."""
    svc = get_inference_service()
    return jsonify({"status": "ok", "model_loaded": svc.is_ready()}), 200
