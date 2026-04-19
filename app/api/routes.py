"""
app/api/routes.py
─────────────────
URL endpoints only. Routes validate → call service → return response.
No business logic, no model code, no image processing.
"""

from flask import Blueprint, request, jsonify, render_template
import json

from app.api.validators import validate_upload, ValidationError
from app.services       import get_inference_service

api_blueprint = Blueprint("api", __name__)

# ── In-memory storage for demo (replace with database in production) ──
analyses_storage = []
users_storage = {}


@api_blueprint.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@api_blueprint.route("/result", methods=["GET"])
def result_page():
    return render_template("result.html")


@api_blueprint.route("/batch", methods=["GET"])
def batch_page():
    """Batch upload interface"""
    return render_template("batch.html")


@api_blueprint.route("/dashboard", methods=["GET"])
def dashboard_page():
    """Analytics dashboard"""
    return render_template("dashboard.html")


@api_blueprint.route("/auth/login", methods=["GET"])
def login_page():
    """Login page"""
    return render_template("login.html")


@api_blueprint.route("/auth/signup", methods=["GET"])
def signup_page():
    """Sign up page"""
    return render_template("signup.html")


# ── Single Prediction ─────────────────────────────────────────────
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
        # Store analysis
        analyses_storage.append({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "label": result.get("label"),
            "score": result.get("score"),
            "confidence": result.get("confidence"),
        })
        return jsonify({"status": "success", **result}), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── Batch Processing ──────────────────────────────────────────────
@api_blueprint.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    """
    POST /api/predict-batch  —  multiple files
    Returns JSON: {status, results: [{label, score, ...}, ...]}
    """
    if "files" not in request.files or len(request.files.getlist("files")) == 0:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    results = []
    errors = []

    for idx, file in enumerate(files):
        try:
            # Validate each file
            image_bytes = file.read()
            if not image_bytes or len(image_bytes) > 16 * 1024 * 1024:
                errors.append({"file": file.filename, "error": "File too large"})
                continue

            # Mock validation (in real app, use validators.py)
            if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                errors.append({"file": file.filename, "error": "Invalid file type"})
                continue

            # Get prediction
            result = get_inference_service().predict(image_bytes)
            result["filename"] = file.filename
            results.append(result)

            # Store analysis
            analyses_storage.append({
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "filename": file.filename,
                "label": result.get("label"),
                "score": result.get("score"),
                "confidence": result.get("confidence"),
            })

        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})

    return jsonify({
        "status": "success",
        "results": results,
        "errors": errors,
        "processed": len(results),
        "failed": len(errors),
    }), 200


# ── Dashboard Statistics ──────────────────────────────────────────
@api_blueprint.route("/api/dashboard/stats", methods=["GET"])
def dashboard_stats():
    """Get dashboard statistics"""
    total = len(analyses_storage)
    normal_count = sum(1 for a in analyses_storage if a.get("label") == "NORMAL")
    anomaly_count = total - normal_count

    avg_confidence = (
        sum(a.get("confidence", 0) for a in analyses_storage) / total
        if total > 0 else 0
    )

    return jsonify({
        "total": total,
        "normal": normal_count,
        "anomalies": anomaly_count,
        "avg_confidence": round(avg_confidence, 2),
        "normal_percentage": round((normal_count / total * 100), 1) if total > 0 else 0,
        "anomaly_percentage": round((anomaly_count / total * 100), 1) if total > 0 else 0,
    }), 200


@api_blueprint.route("/api/dashboard/analyses", methods=["GET"])
def dashboard_analyses():
    """Get recent analyses"""
    limit = request.args.get("limit", 10, type=int)
    return jsonify({
        "analyses": sorted(
            analyses_storage,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )[:limit]
    }), 200


# ── Authentication (Demo) ─────────────────────────────────────────
@api_blueprint.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    """Demo signup endpoint"""
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    name = data.get("name")

    if not all([email, password, name]) or len(password) < 8:
        return jsonify({"error": "Invalid input"}), 400

    if email in users_storage:
        return jsonify({"error": "Email already registered"}), 409

    # Simple password hashing (use bcrypt in production)
    import hashlib
    users_storage[email] = {
        "name": name,
        "password": hashlib.sha256(password.encode()).hexdigest(),
    }

    return jsonify({
        "status": "success",
        "message": "Account created successfully. Please login.",
    }), 201


@api_blueprint.route("/api/auth/login", methods=["POST"])
def auth_login():
    """Demo login endpoint"""
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Missing credentials"}), 400

    if email not in users_storage:
        return jsonify({"error": "Invalid credentials"}), 401

    # Simple password verification (use bcrypt in production)
    import hashlib
    stored_hash = users_storage[email].get("password")
    if hashlib.sha256(password.encode()).hexdigest() != stored_hash:
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({
        "status": "success",
        "message": f"Welcome back, {users_storage[email]['name']}!",
        "token": "demo-jwt-token",
    }), 200


# ── Health Check ──────────────────────────────────────────────────
@api_blueprint.route("/api/health", methods=["GET"])
def health():
    """GET /api/health — liveness check."""
    svc = get_inference_service()
    return jsonify({"status": "ok", "model_loaded": svc.is_ready()}), 200
