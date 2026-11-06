import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
SUPPORTED_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
MAX_DIMENSION = 2048


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attempts."""
    if not filename or not str(filename).strip():
        return "uploaded_image.png"
    clean_path = str(filename).replace("\\", "/")
    return Path(clean_path).name


def validate_image(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Validate image file size, extension, MIME, and structural integrity.
    Rejects corrupted images, oversized files, and unsupported formats.
    """
    if not file_bytes:
        return {"valid": False, "error": "Image file is empty."}

    size_bytes = len(file_bytes)
    if size_bytes > MAX_IMAGE_SIZE_BYTES:
        return {"valid": False, "error": f"Image size ({size_bytes / (1024*1024):.1f} MB) exceeds 10 MB limit."}

    safe_name = sanitize_filename(filename)
    ext = Path(safe_name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return {"valid": False, "error": f"Unsupported image format '{ext}'. Allowed: PNG, JPG, JPEG, WEBP, GIF."}

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()

        # Re-open after verify to fetch dimensions and format
        with Image.open(io.BytesIO(file_bytes)) as img:
            img_format = img.format
            width, height = img.size
            mime = Image.MIME.get(img_format, f"image/{ext.replace('.', '')}")

        if mime not in SUPPORTED_MIMES and f"image/{img_format.lower()}" not in SUPPORTED_MIMES:
            return {"valid": False, "error": f"MIME type mismatch or unsupported format: {mime}."}

        return {
            "valid": True,
            "filename": safe_name,
            "extension": ext,
            "mime_type": mime,
            "size_bytes": size_bytes,
            "width": width,
            "height": height
        }
    except Exception as e:
        return {"valid": False, "error": f"Invalid or corrupted image file: {e}"}


def preprocess_image(file_bytes: bytes) -> Image.Image:
    """
    Preprocess image for Gemini multimodal reasoning:
    - Decode image safely into memory
    - Normalize mode (RGB / RGBA)
    - Downscale giant images (>2048px) maintaining aspect ratio
    """
    img = Image.open(io.BytesIO(file_bytes))
    img.load()

    # Normalize color mode
    if img.mode in ("P", "CMYK", "1", "LA"):
        img = img.convert("RGB")

    # Downscale if exceeding max dimension while preserving quality & aspect ratio
    w, h = img.size
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        if w >= h:
            new_w = MAX_DIMENSION
            new_h = int(h * (MAX_DIMENSION / float(w)))
        else:
            new_h = MAX_DIMENSION
            new_w = int(w * (MAX_DIMENSION / float(h)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def save_temp_image(file_bytes: bytes, filename: str, temp_dir: Optional[str] = None) -> str:
    """Safely save image bytes to a temporary file path."""
    safe_name = sanitize_filename(filename)
    ext = Path(safe_name).suffix.lower() or ".png"

    with tempfile.NamedTemporaryFile("wb", suffix=ext, delete=False, dir=temp_dir) as tmp:
        tmp.write(file_bytes)
        return tmp.name


def cleanup_temp_image(temp_path: Optional[str]) -> None:
    """Clean up temporary image file if it exists."""
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass
