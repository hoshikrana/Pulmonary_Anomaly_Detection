"""
scripts/run_app.py — Flask web app entry point.

Run:  python scripts/run_app.py
Prod: gunicorn "app:create_app()" --bind 0.0.0.0:5000 --workers 1
"""

import os
import sys

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import create_app

if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )
