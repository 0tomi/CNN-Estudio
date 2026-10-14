"""Bounded image decode and frozen-run preprocessing/inference."""
import io
import warnings
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from .data import normalize_images

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def decode_image(content):
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Subí una imagen de hasta 10 MB, no un archivo vacío.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in ("PNG", "JPEG", "WEBP"):
                    raise ValueError("Formato no admitido. Usá PNG, JPEG o WebP.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("La imagen supera 16 millones de píxeles. Reducí su resolución.")
                image.load()
                original = ImageOps.exif_transpose(image).convert("RGBA")
                # Transparent pixels are composited over white, consistently for all datasets.
                background = Image.new("RGBA", original.size, (255, 255, 255, 255))
                return Image.alpha_composite(background, original).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("No se pudo leer la imagen; puede estar corrupta o ser demasiado grande. Usá PNG, JPEG o WebP válido.") from error


def preprocess_image(content, data):
    original = decode_image(content)
    channels, height, width = data.images.shape[1:]
    resized = original.convert("RGB" if channels == 3 else "L").resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.
    raw = torch.from_numpy(array.copy())
    raw = raw.permute(2, 0, 1) if channels == 3 else raw.unsqueeze(0)
    normalized = normalize_images(raw.unsqueeze(0), data.mean, data.std)
    return original, raw, normalized


def classify_latest(runs, content):
    if not runs:
        raise ValueError("Primero entrená una red. Se usará el último entrenamiento completado.")
    run = runs[-1]
    original, raw, normalized = preprocess_image(content, run["data"])
    model = run["model"]
    training = model.training
    try:
        model.eval()
        with torch.no_grad():
            probabilities = model(normalized).softmax(1)[0].cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("Predicción no finita. Revisá el entrenamiento.")
        return {"original": original, "input": raw, "probabilities": probabilities,
                "prediction": int(probabilities.argmax()), "names": run["data"].names,
                "config": run["config"], "best_epoch": run["best_epoch"], "run_number": len(runs)}
    finally:
        model.train(training)
