"""
Screen renderers for LCD display.

Each function returns (Image, hit_dict) where hit_dict maps
button names to (x0, y0, x1, y1) bounding boxes for touch handling.
"""

import time
from typing import Dict, Optional, Tuple, Any, List

from PIL import Image, ImageDraw

from .lcd_driver import WIDTH, HEIGHT
from .lcd_drawing import (
    C_BG, C_ACCENT, C_ACCENT2, C_TEXT, C_SUBTEXT, C_BTN, C_BTN_HI,
    C_PANEL, C_OK, C_WARN,
    FONT_SMALL, FONT_MED, FONT_LARGE, FONT_XLARGE,
    text_size, rounded_rect, centered_text, blank_canvas,
    draw_realtime_clock, draw_posture_icon, draw_progress_bar,
    load_wallpaper,
)

# ── Render caches ──
_dimmed_wallpaper_cache: Dict[int, Image.Image] = {}  # id(wallpaper_img) -> dimmed copy
_thumbnail_cache: Dict[Tuple[str, int, int], Image.Image] = {}  # (path, w, h) -> resized
_timer_done_ring_cache: Optional[Image.Image] = None  # pre-rendered RGBA ring overlay


def invalidate_caches() -> None:
    """Clear all render caches (call when wallpaper changes)."""
    global _timer_done_ring_cache
    _dimmed_wallpaper_cache.clear()
    _thumbnail_cache.clear()
    _timer_done_ring_cache = None


def render_home(
    posture_state: str,
    focus_state: str,
    hydration_status: Dict[str, Any],
    timer_status: Optional[Dict[str, Any]],
    wallpaper_img: Optional[Image.Image],
    notif_pending: bool,
) -> Tuple[Image.Image, Dict]:
    """
    Render the home dashboard screen.

    Shows posture icon, focus state, hydration progress, and timer status.
    """
    if wallpaper_img:
        wp_id = id(wallpaper_img)
        if wp_id not in _dimmed_wallpaper_cache:
            # Build dimmed wallpaper once and cache it
            base = wallpaper_img.copy()
            overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 140))
            base.paste(Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0)),
                       mask=overlay.split()[3])
            _dimmed_wallpaper_cache.clear()  # only keep one cached
            _dimmed_wallpaper_cache[wp_id] = base
        img = _dimmed_wallpaper_cache[wp_id].copy()
        d = ImageDraw.Draw(img)
    else:
        img, d = blank_canvas()

    draw_realtime_clock(d)

    # ── Posture icon + label (top-left) ──
    draw_posture_icon(d, 8, 8, posture_state)
    posture_color = C_OK if posture_state == "good" else C_ACCENT2 if posture_state == "bad" else C_SUBTEXT
    d.text((62, 12), posture_state.upper(), font=FONT_MED, fill=posture_color)

    # ── Focus state ──
    focus_color = C_OK if focus_state == "focused" else C_ACCENT2 if focus_state == "distracted" else C_SUBTEXT
    d.text((10, 68), f"Focus: {focus_state.upper()}", font=FONT_SMALL, fill=focus_color)

    # ── Hydration progress ──
    intake = hydration_status.get("intake_ml", 0)
    goal = hydration_status.get("goal_ml", 2000)
    percent = hydration_status.get("percent", 0)

    d.text((10, 90), f"Water: {intake:.0f}/{goal:.0f} mL  ({percent:.0f}%)",
           font=FONT_SMALL, fill=C_TEXT)
    bar_color = C_OK if percent >= 100 else C_ACCENT if percent >= 50 else C_WARN
    draw_progress_bar(d, 10, 110, WIDTH - 80, 10, percent / 100, fill_color=bar_color)

    last_sip = hydration_status.get("last_sip_time")
    if last_sip:
        ago = int(time.time() - last_sip)
        if ago < 60:
            sip_text = "just now"
        elif ago < 3600:
            sip_text = f"{ago // 60} min ago"
        else:
            sip_text = f"{ago // 3600}h ago"
        d.text((10, 124), f"Last sip: {sip_text}", font=FONT_SMALL, fill=C_SUBTEXT)

    # ── Timer status (if active) ──
    y_timer = 144
    if timer_status and timer_status.get("active"):
        remaining = timer_status.get("remaining_seconds", 0)
        m = int(remaining) // 60
        s = int(remaining) % 60
        phase = timer_status.get("phase", "focus")
        timer_color = C_ACCENT if phase == "focus" else C_OK
        d.text((10, y_timer), f"Timer: {m:02d}:{s:02d} remaining ({phase})",
               font=FONT_SMALL, fill=timer_color)

    # ── Bottom nav buttons ──
    ICON_W, ICON_H = 70, 38
    btn_y0 = HEIGHT - ICON_H - 4
    btn_y1 = HEIGHT - 4

    # Wallpaper button (bottom-left)
    rounded_rect(d, 4, btn_y0, 4 + ICON_W, btn_y1, fill=C_PANEL, outline=C_ACCENT)
    centered_text(d, 4 + ICON_W // 2, (btn_y0 + btn_y1) // 2, "Wall", FONT_SMALL, fill=C_ACCENT)

    # Water button (bottom-center)
    water_x0 = WIDTH // 2 - ICON_W // 2
    water_x1 = WIDTH // 2 + ICON_W // 2
    rounded_rect(d, water_x0, btn_y0, water_x1, btn_y1, fill=C_PANEL, outline=C_OK)
    centered_text(d, WIDTH // 2, (btn_y0 + btn_y1) // 2, "Water", FONT_SMALL, fill=C_OK)

    # Timer button (bottom-right)
    timer_x0 = WIDTH - ICON_W - 4
    timer_x1 = WIDTH - 4
    rounded_rect(d, timer_x0, btn_y0, timer_x1, btn_y1, fill=C_PANEL, outline=C_ACCENT2)
    centered_text(d, (timer_x0 + timer_x1) // 2, (btn_y0 + btn_y1) // 2,
                  "Timer", FONT_SMALL, fill=C_ACCENT2)

    # Notification badge (top-left corner, above posture icon area)
    hits = {
        "wallpaper": (4, btn_y0, 4 + ICON_W, btn_y1),
        "water": (water_x0, btn_y0, water_x1, btn_y1),
        "timer": (timer_x0, btn_y0, timer_x1, btn_y1),
        "notif": None,
    }

    if notif_pending:
        nx0, ny0 = WIDTH - 90, 24
        nx1, ny1 = WIDTH - 10, 48
        rounded_rect(d, nx0, ny0, nx1, ny1, fill=C_ACCENT2)
        centered_text(d, (nx0 + nx1) // 2, (ny0 + ny1) // 2, "! Notif", FONT_SMALL, fill=C_TEXT)
        hits["notif"] = (nx0, ny0, nx1, ny1)

    return img, hits


def render_notification(message: str, detail: str = "") -> Tuple[Image.Image, Dict]:
    """Render an alert/notification screen with acknowledge button."""
    img, d = blank_canvas()

    px, py = 20, 60
    pw, ph = WIDTH - 40, HEIGHT - 120
    rounded_rect(d, px, py, px + pw, py + ph, radius=14, fill=C_PANEL, outline=C_ACCENT2)

    # Icon ring
    d.ellipse([px + pw // 2 - 28, py + 10, px + pw // 2 + 28, py + 66],
              outline=C_ACCENT2, width=3)
    centered_text(d, px + pw // 2, py + 38, "!", FONT_XLARGE, fill=C_ACCENT2)

    # Word-wrap message
    words = message.split()
    lines, line = [], ""
    max_w = pw - 20
    for w in words:
        test = (line + " " + w).strip()
        tw, _ = text_size(d, test, FONT_MED)
        if tw <= max_w:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)

    ty = py + 80
    for ln in lines:
        centered_text(d, px + pw // 2, ty, ln, FONT_MED, fill=C_TEXT)
        ty += 26

    if detail:
        centered_text(d, px + pw // 2, ty + 10, detail, FONT_SMALL, fill=C_SUBTEXT)

    # Acknowledge button
    bx0, by0 = WIDTH // 2 - 60, HEIGHT - 52
    bx1, by1 = WIDTH // 2 + 60, HEIGHT - 12
    rounded_rect(d, bx0, by0, bx1, by1, fill=C_OK)
    centered_text(d, WIDTH // 2, (by0 + by1) // 2, "Acknowledge", FONT_MED, fill=(0, 0, 0))

    draw_realtime_clock(d)
    return img, {"ack": (bx0, by0, bx1, by1)}


def render_timer_setup(hours: int, minutes: int, seconds: int,
                       editing_field: int) -> Tuple[Image.Image, Dict]:
    """Render timer setup screen with HH:MM:SS fields and touch arrows."""
    img, d = blank_canvas()
    draw_realtime_clock(d)

    centered_text(d, WIDTH // 2, 28, "Set Timer", FONT_LARGE, fill=C_ACCENT)

    labels = ["HH", "MM", "SS"]
    values = [hours, minutes, seconds]
    field_x = [60, WIDTH // 2, WIDTH - 60]
    field_y = HEIGHT // 2

    SEG_W, SEG_H = 70, 56

    for i in range(3):
        cx = field_x[i]
        is_active = (i == editing_field)
        col = C_ACCENT if is_active else C_BTN
        rounded_rect(d, cx - SEG_W // 2, field_y - SEG_H // 2,
                     cx + SEG_W // 2, field_y + SEG_H // 2,
                     fill=col, outline=C_ACCENT if is_active else C_SUBTEXT)
        centered_text(d, cx, field_y - 10, "{:02d}".format(values[i]), FONT_XLARGE,
                      fill=(0, 0, 0) if is_active else C_TEXT)
        centered_text(d, cx, field_y + SEG_H // 2 + 12, labels[i], FONT_SMALL, fill=C_SUBTEXT)

        ax0, ax1 = cx - 20, cx + 20
        # Up arrow
        rounded_rect(d, ax0, field_y - SEG_H // 2 - 36, ax1, field_y - SEG_H // 2 - 6, fill=C_BTN_HI)
        centered_text(d, cx, field_y - SEG_H // 2 - 21, "^", FONT_MED, fill=C_TEXT)
        # Down arrow
        rounded_rect(d, ax0, field_y + SEG_H // 2 + 6, ax1, field_y + SEG_H // 2 + 36, fill=C_BTN_HI)
        centered_text(d, cx, field_y + SEG_H // 2 + 21, "v", FONT_MED, fill=C_TEXT)

        if i < 2:
            mid_x = (field_x[i] + field_x[i + 1]) // 2
            centered_text(d, mid_x, field_y - 6, ":", FONT_XLARGE, fill=C_SUBTEXT)

    # Start / Cancel buttons
    rounded_rect(d, 20, HEIGHT - 44, WIDTH // 2 - 10, HEIGHT - 6, fill=C_OK)
    centered_text(d, WIDTH // 4 + 5, HEIGHT - 25, "Start", FONT_MED, fill=(0, 0, 0))

    rounded_rect(d, WIDTH // 2 + 10, HEIGHT - 44, WIDTH - 20, HEIGHT - 6, fill=C_ACCENT2)
    centered_text(d, 3 * WIDTH // 4 - 5, HEIGHT - 25, "Cancel", FONT_MED, fill=(255, 255, 255))

    hits = {}
    SEG_HALF = SEG_W // 2
    for i in range(3):
        cx = field_x[i]
        ax0, ax1 = cx - 20, cx + 20
        fh = SEG_H // 2
        hits[f"up_{i}"] = (ax0, field_y - fh - 36, ax1, field_y - fh - 6)
        hits[f"down_{i}"] = (ax0, field_y + fh + 6, ax1, field_y + fh + 36)
        hits[f"field_{i}"] = (cx - SEG_HALF, field_y - fh, cx + SEG_HALF, field_y + fh)
    hits["start"] = (20, HEIGHT - 44, WIDTH // 2 - 10, HEIGHT - 6)
    hits["cancel"] = (WIDTH // 2 + 10, HEIGHT - 44, WIDTH - 20, HEIGHT - 6)

    return img, hits


def render_timer_run(remaining_secs: int, total_secs: int) -> Tuple[Image.Image, Dict]:
    """Render timer countdown screen with progress bar."""
    img, d = blank_canvas()
    draw_realtime_clock(d)

    centered_text(d, WIDTH // 2, 26, "Timer", FONT_MED, fill=C_SUBTEXT)

    h = remaining_secs // 3600
    m = (remaining_secs % 3600) // 60
    s = remaining_secs % 60
    timer_str = "{:02d}:{:02d}:{:02d}".format(h, m, s)
    centered_text(d, WIDTH // 2, HEIGHT // 2 - 20, timer_str, FONT_XLARGE, fill=C_ACCENT)

    # Progress bar
    progress = 1.0 - (remaining_secs / max(total_secs, 1))
    draw_progress_bar(d, 30, HEIGHT // 2 + 40, WIDTH - 60, 14, progress)

    # End button
    bx0, by0 = WIDTH // 2 - 60, HEIGHT - 44
    bx1, by1 = WIDTH // 2 + 60, HEIGHT - 6
    rounded_rect(d, bx0, by0, bx1, by1, fill=C_ACCENT2)
    centered_text(d, WIDTH // 2, (by0 + by1) // 2, "End", FONT_MED, fill=(255, 255, 255))

    return img, {"end": (bx0, by0, bx1, by1)}


def render_timer_done() -> Tuple[Image.Image, Dict]:
    """Render timer completion screen with glowing rings."""
    global _timer_done_ring_cache

    img, d = blank_canvas()
    draw_realtime_clock(d)

    cx, cy = WIDTH // 2, HEIGHT // 2 - 20

    # Build ring overlay once and reuse
    if _timer_done_ring_cache is None:
        ring_overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        for r, alpha in [(70, 40), (55, 80), (40, 160)]:
            ring = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            rd = ImageDraw.Draw(ring)
            col = (*C_ACCENT2, alpha)
            rd.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=4)
            ring_overlay = Image.alpha_composite(ring_overlay, ring)
        _timer_done_ring_cache = ring_overlay

    img = Image.alpha_composite(img.convert("RGBA"), _timer_done_ring_cache).convert("RGB")
    d = ImageDraw.Draw(img)

    centered_text(d, WIDTH // 2, cy, "Done!", FONT_LARGE, fill=C_OK)
    centered_text(d, WIDTH // 2, HEIGHT // 2 + 50, "Time's Up!", FONT_LARGE, fill=C_ACCENT2)

    bx0, by0 = WIDTH // 2 - 60, HEIGHT - 44
    bx1, by1 = WIDTH // 2 + 60, HEIGHT - 6
    rounded_rect(d, bx0, by0, bx1, by1, fill=C_OK)
    centered_text(d, WIDTH // 2, (by0 + by1) // 2, "Acknowledge", FONT_MED, fill=(0, 0, 0))

    return img, {"ack": (bx0, by0, bx1, by1)}


def render_wallpaper_picker(current_idx: int,
                            wallpapers: List[str]) -> Tuple[Image.Image, Dict]:
    """Render wallpaper thumbnail grid."""
    img, d = blank_canvas()
    draw_realtime_clock(d)

    centered_text(d, WIDTH // 2, 28, "Choose Wallpaper", FONT_MED, fill=C_ACCENT)

    thumbs_per_row = 3
    thumb_w = (WIDTH - 40) // thumbs_per_row
    thumb_h = 80
    margin = 10
    start_y = 50
    hits = {}

    for i, wp in enumerate(wallpapers):
        col = i % thumbs_per_row
        row = i // thumbs_per_row
        x0 = 10 + col * (thumb_w + margin // 2)
        y0 = start_y + row * (thumb_h + margin)
        x1 = x0 + thumb_w
        y1 = y0 + thumb_h

        try:
            cache_key = (wp, thumb_w, thumb_h)
            if cache_key not in _thumbnail_cache:
                _thumbnail_cache[cache_key] = load_wallpaper(wp).resize(
                    (thumb_w, thumb_h), Image.BICUBIC
                )
            img.paste(_thumbnail_cache[cache_key], (x0, y0))
        except Exception:
            rounded_rect(d, x0, y0, x1, y1, fill=C_BTN)

        outline = C_ACCENT if i == current_idx else C_SUBTEXT
        d.rectangle([x0, y0, x1, y1], outline=outline,
                    width=3 if i == current_idx else 1)
        hits[f"wp_{i}"] = (x0, y0, x1, y1)

    # Back button
    bx0, by0 = WIDTH // 2 - 50, HEIGHT - 44
    bx1, by1 = WIDTH // 2 + 50, HEIGHT - 6
    rounded_rect(d, bx0, by0, bx1, by1, fill=C_BTN, outline=C_ACCENT)
    centered_text(d, WIDTH // 2, (by0 + by1) // 2, "Back", FONT_MED, fill=C_ACCENT)
    hits["back"] = (bx0, by0, bx1, by1)

    return img, hits


def render_water_goal_edit(current_goal_ml: float) -> Tuple[Image.Image, Dict]:
    """Render water goal adjustment screen with +/- buttons."""
    img, d = blank_canvas()
    draw_realtime_clock(d)

    centered_text(d, WIDTH // 2, 28, "Daily Water Goal", FONT_LARGE, fill=C_ACCENT)

    # Display current goal
    centered_text(d, WIDTH // 2, HEIGHT // 2 - 10,
                  f"{int(current_goal_ml)} mL", FONT_XLARGE, fill=C_TEXT)

    # +/- buttons
    btn_w = 60
    btn_h = 44

    # Minus 250
    mx0 = WIDTH // 4 - btn_w // 2
    my0 = HEIGHT // 2 + 30
    rounded_rect(d, mx0, my0, mx0 + btn_w, my0 + btn_h, fill=C_ACCENT2)
    centered_text(d, mx0 + btn_w // 2, my0 + btn_h // 2, "-250", FONT_MED, fill=C_TEXT)

    # Plus 250
    px0 = 3 * WIDTH // 4 - btn_w // 2
    py0 = my0
    rounded_rect(d, px0, py0, px0 + btn_w, py0 + btn_h, fill=C_OK)
    centered_text(d, px0 + btn_w // 2, py0 + btn_h // 2, "+250", FONT_MED, fill=(0, 0, 0))

    # Done button
    bx0, by0 = WIDTH // 2 - 50, HEIGHT - 44
    bx1, by1 = WIDTH // 2 + 50, HEIGHT - 6
    rounded_rect(d, bx0, by0, bx1, by1, fill=C_BTN, outline=C_ACCENT)
    centered_text(d, WIDTH // 2, (by0 + by1) // 2, "Done", FONT_MED, fill=C_ACCENT)

    hits = {
        "minus": (mx0, my0, mx0 + btn_w, my0 + btn_h),
        "plus": (px0, py0, px0 + btn_w, py0 + btn_h),
        "done": (bx0, by0, bx1, by1),
    }
    return img, hits


def render_hydration_detail(hydration_status: Dict[str, Any]) -> Tuple[Image.Image, Dict]:
    """Render full hydration stats screen."""
    img, d = blank_canvas()
    draw_realtime_clock(d)

    centered_text(d, WIDTH // 2, 28, "Hydration", FONT_LARGE, fill=C_ACCENT)

    intake = hydration_status.get("intake_ml", 0)
    goal = hydration_status.get("goal_ml", 2000)
    percent = hydration_status.get("percent", 0)
    cup_name = hydration_status.get("cup_name")

    # Large percentage display
    pct_color = C_OK if percent >= 100 else C_ACCENT if percent >= 50 else C_WARN
    centered_text(d, WIDTH // 2, 70, f"{percent:.0f}%", FONT_XLARGE, fill=pct_color)

    # Progress bar
    draw_progress_bar(d, 30, 100, WIDTH - 60, 16, percent / 100, fill_color=pct_color)

    # Stats
    d.text((20, 130), f"Intake: {intake:.0f} mL", font=FONT_MED, fill=C_TEXT)
    d.text((20, 156), f"Goal:   {goal:.0f} mL", font=FONT_MED, fill=C_SUBTEXT)

    if cup_name:
        d.text((20, 182), f"Cup: {cup_name}", font=FONT_SMALL, fill=C_SUBTEXT)

    # Edit goal button
    gx0, gy0 = 20, HEIGHT - 44
    gx1, gy1 = WIDTH // 2 - 10, HEIGHT - 6
    rounded_rect(d, gx0, gy0, gx1, gy1, fill=C_BTN, outline=C_ACCENT)
    centered_text(d, (gx0 + gx1) // 2, (gy0 + gy1) // 2, "Set Goal", FONT_SMALL, fill=C_ACCENT)

    # Back button
    bx0, by0 = WIDTH // 2 + 10, HEIGHT - 44
    bx1, by1 = WIDTH - 20, HEIGHT - 6
    rounded_rect(d, bx0, by0, bx1, by1, fill=C_BTN, outline=C_ACCENT)
    centered_text(d, (bx0 + bx1) // 2, (by0 + by1) // 2, "Back", FONT_SMALL, fill=C_ACCENT)

    hits = {
        "set_goal": (gx0, gy0, gx1, gy1),
        "back": (bx0, by0, bx1, by1),
    }
    return img, hits
