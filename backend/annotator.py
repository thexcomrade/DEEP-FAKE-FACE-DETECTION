"""
Annotator module for Deepfake Detection result images.

Handles safe drawing, dynamic font scaling, overlay positioning,
and format preservation for result images.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageDraw, ImageFont, ImageOps

logger = logging.getLogger(__name__)


def _get_font(font_size: int) -> ImageFont.ImageFont:
    """Attempt to load a standard scalable TTF font, falling back to default."""
    font_candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "SegoeUI.ttf",
        "Helvetica.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def generate_result_image(
    image_path: Union[str, Path],
    result: str,
    confidence: float,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """
    Generate an annotated result image displaying the prediction and confidence score.
    
    Dynamically scales text, font, padding, and positioning according to image dimensions.
    Returns the absolute path to the generated image, or None on failure.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error("Source image not found for annotation: %s", image_path)
        return None

    try:
        with Image.open(image_path) as orig_img:
            # Normalize orientation from EXIF if present
            img = ImageOps.exif_transpose(orig_img)
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            width, height = img.size
            if width < 10 or height < 10:
                logger.error("Image too small to annotate (%dx%d)", width, height)
                return None

            # Dynamic font size: 4.5% to 6% of min dimension, bounded reasonably
            min_dim = min(width, height)
            font_size = max(14, min(72, int(min_dim * 0.055)))
            font = _get_font(font_size)

            label_text = f"{result} ({confidence:.1f}%)"

            # Create dummy draw to measure text
            dummy_draw = ImageDraw.Draw(img)
            try:
                bbox = dummy_draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except Exception:
                # Fallback for very old Pillow
                text_width = font_size * len(label_text) * 0.6
                text_height = font_size

            # Position near bottom with margin
            pad_x = max(10, int(font_size * 0.6))
            pad_y = max(6, int(font_size * 0.35))
            bottom_margin = max(15, int(height * 0.05))

            center_x = width // 2
            center_y = height - bottom_margin - (text_height // 2)

            box_left = max(0, center_x - (text_width // 2) - pad_x)
            box_right = min(width, center_x + (text_width // 2) + pad_x)
            box_top = max(0, center_y - (text_height // 2) - pad_y)
            box_bottom = min(height, center_y + (text_height // 2) + pad_y)

            # Translucent dark pill overlay
            overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            radius = max(4, int(font_size * 0.3))
            try:
                overlay_draw.rounded_rectangle(
                    [box_left, box_top, box_right, box_bottom],
                    radius=radius,
                    fill=(15, 15, 15, 180),
                )
            except AttributeError:
                overlay_draw.rectangle(
                    [box_left, box_top, box_right, box_bottom],
                    fill=(15, 15, 15, 180),
                )

            # Composite overlay
            annotated = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(annotated)

            # Colors: Green for Real, Red for Fake
            text_color = "#28a745" if result == "Real" else "#dc3545"
            draw.text(
                (center_x, center_y),
                label_text,
                fill=text_color,
                font=font,
                anchor="mm",
            )

            # Determine output destination
            if output_path is None:
                parent = image_path.parent
                filename = f"result_{image_path.name}"
                output_path = parent / filename
            else:
                output_path = Path(output_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as RGB JPEG if extension is .jpg/.jpeg, else PNG
            out_ext = output_path.suffix.lower()
            if out_ext in [".jpg", ".jpeg"]:
                final_rgb = annotated.convert("RGB")
                final_rgb.save(str(output_path), "JPEG", quality=92, optimize=True)
            else:
                annotated.save(str(output_path), optimize=True)

            logger.info("Generated annotated result image at %s", output_path)
            return str(output_path)

    except Exception as exc:
        logger.error("Failed to generate annotated image for %s: %s", image_path, exc)
        return None
