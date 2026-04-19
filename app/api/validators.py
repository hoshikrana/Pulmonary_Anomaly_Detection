"""
app/api/validators.py
─────────────────────
Server-side request validation. Single responsibility: inspect the
request and raise ValidationError before any inference happens.
"""

from flask import Request
import config

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


class ValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message     = message
        self.status_code = status_code


def validate_upload(request: Request) -> bytes:
    """
    Validate a multipart file upload.

    Checks: field present, filename not empty, extension allowed, size within limit.
    Returns raw image bytes on success, raises ValidationError otherwise.
    """
    if "file" not in request.files:
        raise ValidationError(
            "No 'file' field in request. Send as multipart/form-data.", 400)

    file = request.files["file"]

    if not file.filename:
        raise ValidationError("No file selected.", 400)

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file type '.{ext}'. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}.", 415)

    image_bytes = file.read()

    if len(image_bytes) == 0:
        raise ValidationError("Uploaded file is empty.", 400)

    if len(image_bytes) > config.MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"File too large ({len(image_bytes)/1e6:.1f} MB). "
            f"Maximum is {config.MAX_UPLOAD_BYTES // (1024*1024)} MB.", 413)

    return image_bytes
