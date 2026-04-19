"""
app/__init__.py — Flask application factory.

create_app() is the single entry point.
MAX_CONTENT_LENGTH pulled from config (single source of truth).
"""

import os
from flask import Flask
import config


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES
    app.config["UPLOAD_FOLDER"]      = os.path.join(config.OUTPUT_DIR, "uploads")
    app.config["SECRET_KEY"]         = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
    app.config["DEBUG"]              = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    from app.api.routes import api_blueprint
    app.register_blueprint(api_blueprint)

    _register_error_handlers(app)
    return app


def _register_error_handlers(app: Flask) -> None:
    from flask import render_template, jsonify, request

    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Endpoint not found."}), 404
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": f"File too large. Max {config.MAX_UPLOAD_BYTES//(1024*1024)} MB."}), 413

    @app.errorhandler(500)
    def server_error(e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal server error."}), 500
        return render_template("error.html", code=500, message="Something went wrong."), 500
