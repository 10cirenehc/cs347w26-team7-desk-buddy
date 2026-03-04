"""
Drawing utilities for LCD screens.

Ported from lcd.py with additions for posture icons and hydration visuals.
"""

import os
from typing import Tuple, Optional

from PIL import Image, ImageDraw, ImageFont

from .lcd_driver import WIDTH, HEIGHT

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_BG = (10, 10, 20)
C_ACCENT = (0, 180, 255)
C_ACCENT2 = (255, 80, 120)
C_TEXT = (230, 235, 255)
C_SUBTEXT = (120, 130, 160)
C_BTN = (30, 40, 70)
C_BTN_HI = (50, 70, 120)
C_PANEL = (20, 25, 50)
C_OK = (0, 200, 120)
C_WARN = (255, 160, 30)

# ─────────────────────────────────────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────────────────────────────────────
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a TrueType font, falling back to default."""
    for path in _FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_SMALL = load_font(14)
FONT_MED = load_font(20)
FONT_LARGE = load_font(28)
FONT_XLARGE = load_font(42)

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def text_size(draw: ImageDraw.Draw, text: str, font) -> Tuple[int, int]:
    """Return (width, height) of text — works on both old and new Pillow."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def rounded_rect(draw: ImageDraw.Draw, x0, y0, x1, y1,
                 radius=10, fill=C_BTN, outline=None):
    """Draw a rounded rectangle compatible with Pillow < 8.2.0."""
    r = min(radius, (x1 - x0) // 2, (y1 - y0) // 2)
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
    draw.ellipse([x0, y0, x0 + 2 * r, y0 + 2 * r], fill=fill)
    draw.ellipse([x1 - 2 * r, y0, x1, y0 + 2 * r], fill=fill)
    draw.ellipse([x0, y1 - 2 * r, x0 + 2 * r, y1], fill=fill)
    draw.ellipse([x1 - 2 * r, y1 - 2 * r, x1, y1], fill=fill)
    if outline:
        draw.arc([x0, y0, x0 + 2 * r, y0 + 2 * r], 180, 270, fill=outline)
        draw.arc([x1 - 2 * r, y0, x1, y0 + 2 * r], 270, 360, fill=outline)
        draw.arc([x0, y1 - 2 * r, x0 + 2 * r, y1], 90, 180, fill=outline)
        draw.arc([x1 - 2 * r, y1 - 2 * r, x1, y1], 0, 90, fill=outline)
        draw.line([x0 + r, y0, x1 - r, y0], fill=outline)
        draw.line([x0 + r, y1, x1 - r, y1], fill=outline)
        draw.line([x0, y0 + r, x0, y1 - r], fill=outline)
        draw.line([x1, y0 + r, x1, y1 - r], fill=outline)


def centered_text(draw: ImageDraw.Draw, cx, cy, text, font, fill=C_TEXT):
    """Draw text centered at (cx, cy)."""
    tw, th = text_size(draw, text, font)
    draw.text((cx - tw // 2, cy - th // 2), text, font=font, fill=fill)


def blank_canvas() -> Tuple[Image.Image, ImageDraw.Draw]:
    """Create a blank image with background color."""
    img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    d = ImageDraw.Draw(img)
    return img, d


def touch_in(tx, ty, x0, y0, x1, y1) -> bool:
    """Check if touch coordinates are inside a bounding box."""
    return x0 <= tx <= x1 and y0 <= ty <= y1


def draw_realtime_clock(draw: ImageDraw.Draw):
    """Draw current time in top-right corner."""
    import time
    t = time.localtime()
    ts = "{:02d}:{:02d}:{:02d}".format(t.tm_hour, t.tm_min, t.tm_sec)
    tw, _ = text_size(draw, ts, FONT_SMALL)
    draw.text((WIDTH - tw - 6, 5), ts, font=FONT_SMALL, fill=C_ACCENT)


def draw_posture_icon(draw: ImageDraw.Draw, x: int, y: int, state: str):
    """
    Draw a stick figure posture icon (~50x60px).

    Args:
        draw: ImageDraw instance
        x: Top-left x coordinate
        y: Top-left y coordinate
        state: "good", "bad", or "unknown"
    """
    if state == "good":
        color = C_OK
        # Upright figure
        head_cx, head_cy = x + 25, y + 10
        draw.ellipse([head_cx - 6, head_cy - 6, head_cx + 6, head_cy + 6],
                     outline=color, width=2)
        # Straight spine
        draw.line([head_cx, head_cy + 6, head_cx, y + 40], fill=color, width=2)
        # Arms (slightly down)
        draw.line([head_cx, y + 22, head_cx - 14, y + 32], fill=color, width=2)
        draw.line([head_cx, y + 22, head_cx + 14, y + 32], fill=color, width=2)
        # Legs
        draw.line([head_cx, y + 40, head_cx - 10, y + 56], fill=color, width=2)
        draw.line([head_cx, y + 40, head_cx + 10, y + 56], fill=color, width=2)

    elif state == "bad":
        color = C_ACCENT2
        # Slouched figure
        head_cx, head_cy = x + 30, y + 12
        draw.ellipse([head_cx - 6, head_cy - 6, head_cx + 6, head_cy + 6],
                     outline=color, width=2)
        # Curved spine (forward lean)
        draw.line([head_cx, head_cy + 6, head_cx - 3, y + 25], fill=color, width=2)
        draw.line([head_cx - 3, y + 25, head_cx - 5, y + 40], fill=color, width=2)
        # Arms (drooping)
        draw.line([head_cx - 3, y + 22, head_cx - 16, y + 34], fill=color, width=2)
        draw.line([head_cx - 3, y + 22, head_cx + 10, y + 34], fill=color, width=2)
        # Legs
        draw.line([head_cx - 5, y + 40, head_cx - 14, y + 56], fill=color, width=2)
        draw.line([head_cx - 5, y + 40, head_cx + 6, y + 56], fill=color, width=2)

    else:
        color = C_SUBTEXT
        # Neutral figure
        head_cx, head_cy = x + 25, y + 10
        draw.ellipse([head_cx - 6, head_cy - 6, head_cx + 6, head_cy + 6],
                     outline=color, width=2)
        draw.line([head_cx, head_cy + 6, head_cx, y + 40], fill=color, width=2)
        draw.line([head_cx, y + 22, head_cx - 14, y + 32], fill=color, width=2)
        draw.line([head_cx, y + 22, head_cx + 14, y + 32], fill=color, width=2)
        draw.line([head_cx, y + 40, head_cx - 10, y + 56], fill=color, width=2)
        draw.line([head_cx, y + 40, head_cx + 10, y + 56], fill=color, width=2)


def draw_progress_bar(draw: ImageDraw.Draw, x, y, w, h, progress: float,
                      fill_color=C_OK, bg_color=C_BTN):
    """Draw a horizontal progress bar (0.0 to 1.0)."""
    progress = max(0.0, min(1.0, progress))
    rounded_rect(draw, x, y, x + w, y + h, radius=h // 2, fill=bg_color)
    fill_w = int(w * progress)
    if fill_w > 4:
        rounded_rect(draw, x, y, x + fill_w, y + h, radius=h // 2, fill=fill_color)


# ─────────────────────────────────────────────────────────────────────────────
# WALLPAPER LOADING
# ─────────────────────────────────────────────────────────────────────────────
_wallpaper_cache = {}


def load_wallpaper(path: str) -> Image.Image:
    """Load and crop a wallpaper image to fit the display."""
    if path in _wallpaper_cache:
        return _wallpaper_cache[path]
    try:
        img = Image.open(path)
        ratio = img.width / img.height
        sratio = WIDTH / HEIGHT
        if sratio < ratio:
            sw, sh = img.width * HEIGHT // img.height, HEIGHT
        else:
            sw, sh = WIDTH, img.height * WIDTH // img.width
        img = img.resize((sw, sh), Image.BICUBIC)
        cx = sw // 2 - WIDTH // 2
        cy = sh // 2 - HEIGHT // 2
        img = img.crop((cx, cy, cx + WIDTH, cy + HEIGHT)).convert("RGB")
        _wallpaper_cache[path] = img
        return img
    except Exception:
        img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
        _wallpaper_cache[path] = img
        return img


def find_wallpapers(wallpaper_dir: str) -> list:
    """Find wallpaper image files in a directory."""
    wallpapers = []
    if os.path.isdir(wallpaper_dir):
        for f in sorted(os.listdir(wallpaper_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                wallpapers.append(os.path.join(wallpaper_dir, f))
    return wallpapers
