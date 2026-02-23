# Katie's UI/UX edits
import digitalio
import board
import busio
import time
import os
from PIL import Image, ImageDraw, ImageFont
import adafruit_rgb_display.ili9341 as ili9341
import adafruit_focaltouch

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY SETUP
# ─────────────────────────────────────────────────────────────────────────────
cs_pin    = digitalio.DigitalInOut(board.CE0)
dc_pin    = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D24)
BAUDRATE  = 24000000

spi  = board.SPI()
i2c  = busio.I2C(board.SCL_1, board.SDA_1)
ft   = adafruit_focaltouch.Adafruit_FocalTouch(i2c, debug=False)
disp = ili9341.ILI9341(spi, rotation=90,
                        cs=cs_pin, dc=dc_pin, rst=reset_pin, baudrate=BAUDRATE)

if disp.rotation % 180 == 90:
    HEIGHT, WIDTH = disp.width, disp.height
else:
    WIDTH, HEIGHT = disp.width, disp.height

# ─────────────────────────────────────────────────────────────────────────────
# FONTS  (falls back gracefully if truetype not present)
# ─────────────────────────────────────────────────────────────────────────────
def load_font(size):
    for path in [
        "/usr/share/fonts/truetype/Gargi/Gargi.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

FONT_SMALL  = load_font(14)
FONT_MED    = load_font(20)
FONT_LARGE  = load_font(28)
FONT_XLARGE = load_font(42)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_BG        = (240,  240,  240) # Cream 
C_ACCENT    = (184,  167, 255)  # Purple
C_ACCENT2   = (125, 218,  146)  # Green
C_TEXT      = (61, 61, 61)      # Dark Gray
C_SUBTEXT   = (120, 130, 160)
C_BTN       = (30,  40,  70)
C_BTN_HI    = (50,  70, 120)
C_PANEL     = (20,  25,  50)
C_OK        = (0,   200, 120)
C_WARN      = (254, 84,  24)    # Reddish

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def now_str():
    t = time.localtime()
    return "{:02d}:{:02d}:{:02d}".format(t.tm_hour, t.tm_min, t.tm_sec)

def text_size(draw, text, font):
    """Return (width, height) of text — works on both old and new Pillow."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Pillow < 8.0.0 uses textsize instead
        return draw.textsize(text, font=font)

def draw_realtime_clock(draw):
    """Stamp current time top-right, always visible."""
    ts = now_str()
    tw, _ = text_size(draw, ts, FONT_SMALL)
    draw.text((WIDTH - tw - 6, 5), ts, font=FONT_SMALL, fill=C_ACCENT)

def rounded_rect(draw, x0, y0, x1, y1, radius=10, fill=C_BTN, outline=None):
    """Draw a rounded rectangle compatible with Pillow < 8.2.0."""
    r = min(radius, (x1 - x0) // 2, (y1 - y0) // 2)
    # Fill the three rects that make the cross shape
    draw.rectangle([x0 + r, y0,     x1 - r, y1    ], fill=fill)
    draw.rectangle([x0,     y0 + r, x1,     y1 - r], fill=fill)
    # Fill the four corner circles
    draw.ellipse([x0,     y0,     x0+2*r, y0+2*r], fill=fill)
    draw.ellipse([x1-2*r, y0,     x1,     y0+2*r], fill=fill)
    draw.ellipse([x0,     y1-2*r, x0+2*r, y1    ], fill=fill)
    draw.ellipse([x1-2*r, y1-2*r, x1,     y1    ], fill=fill)
    # Outline
    if outline:
        draw.arc([x0,     y0,     x0+2*r, y0+2*r], 180, 270, fill=outline)
        draw.arc([x1-2*r, y0,     x1,     y0+2*r], 270, 360, fill=outline)
        draw.arc([x0,     y1-2*r, x0+2*r, y1    ], 90,  180, fill=outline)
        draw.arc([x1-2*r, y1-2*r, x1,     y1    ], 0,   90,  fill=outline)
        draw.line([x0+r, y0,  x1-r, y0 ], fill=outline)  # top
        draw.line([x0+r, y1,  x1-r, y1 ], fill=outline)  # bottom
        draw.line([x0,   y0+r, x0,  y1-r], fill=outline)  # left
        draw.line([x1,   y0+r, x1,  y1-r], fill=outline)  # right

def centered_text(draw, cx, cy, text, font, fill=C_TEXT):
    tw, th = text_size(draw, text, font)
    draw.text((cx - tw // 2, cy - th // 2), text, font=font, fill=fill)

def push(image):
    disp.image(image)

def blank_canvas():
    img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    d   = ImageDraw.Draw(img)
    return img, d

def touch_in(tx, ty, x0, y0, x1, y1):
    return x0 <= tx <= x1 and y0 <= ty <= y1

def get_touch():
    """Return (x, y) of first touch in display coords, or None."""
    if ft.touched:
        touches = ft.touches
        if touches and len(touches) > 0:
            t = touches[0]
            # FocalTouch reports in sensor coords; ILI9341 rotated 90° landscape:
            # raw x → display y,  raw y → display x
            raw_x = t["x"]
            raw_y = t["y"]
            # sensor is 240×320, display is 320×240 after rotation
            disp_x = raw_y
            disp_y = raw_x
            return disp_x, disp_y
    return None

# ─────────────────────────────────────────────────────────────────────────────
# WALLPAPER MANAGER
# ─────────────────────────────────────────────────────────────────────────────
WALLPAPERS = ["shark.jpg"]          # populate with your image files
for f in os.listdir("."):
    if f.lower().endswith((".jpg", ".jpeg", ".png")) and f not in WALLPAPERS:
        WALLPAPERS.append(f)
WALLPAPERS = list(dict.fromkeys(WALLPAPERS))  # deduplicate

_wallpaper_cache = {}

def load_wallpaper(path):
    if path in _wallpaper_cache:
        return _wallpaper_cache[path]
    try:
        img = Image.open(path)
        ratio  = img.width / img.height
        sratio = WIDTH / HEIGHT
        if sratio < ratio:
            sw, sh = img.width * HEIGHT // img.height, HEIGHT
        else:
            sw, sh = WIDTH, img.height * WIDTH // img.width
        img = img.resize((sw, sh), Image.BICUBIC)
        x = sw // 2 - WIDTH  // 2
        y = sh // 2 - HEIGHT // 2
        img = img.crop((x, y, x + WIDTH, y + HEIGHT)).convert("RGB")
        _wallpaper_cache[path] = img
        return img
    except Exception:
        img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
        _wallpaper_cache[path] = img
        return img

# ─────────────────────────────────────────────────────────────────────────────
# STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────
STATE_HOME          = "home"
STATE_NOTIFICATION  = "notification"
STATE_TIMER_SETUP   = "timer_setup"
STATE_TIMER_RUN     = "timer_run"
STATE_TIMER_DONE    = "timer_done"
STATE_WALLPAPER     = "wallpaper"

# ─────────────────────────────────────────────────────────────────────────────
# HOME SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def draw_home(wallpaper_idx, notification=None):
    img = load_wallpaper(WALLPAPERS[wallpaper_idx]).copy()
    d   = ImageDraw.Draw(img)

    # dim overlay so buttons are readable
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 120))
    img.paste(Image.new("RGB", (WIDTH, HEIGHT), (0,0,0)),
              mask=overlay.split()[3])

    # Clock top-right
    draw_realtime_clock(d)

    # ── corner icons (60×50 buttons) ───────────────────────────────────────
    ICON_W, ICON_H = 60, 44

    # Bottom-left  → wallpaper picker
    rounded_rect(d, 4, HEIGHT-ICON_H-4, 4+ICON_W, HEIGHT-4, fill=C_PANEL, outline=C_ACCENT)
    centered_text(d, 4+ICON_W//2, HEIGHT-ICON_H//2-4, "🖼 Wall", FONT_SMALL, fill=C_ACCENT)

    # Bottom-right → timer setup
    rounded_rect(d, WIDTH-ICON_W-4, HEIGHT-ICON_H-4, WIDTH-4, HEIGHT-4, fill=C_PANEL, outline=C_ACCENT2)
    centered_text(d, WIDTH-ICON_W//2-4, HEIGHT-ICON_H//2-4, "⏱ Timer", FONT_SMALL, fill=C_ACCENT2)

    # Notification badge (top-left) if pending
    if notification:
        rounded_rect(d, 4, 4, 4+ICON_W+20, 4+ICON_H, fill=C_ACCENT2, outline=None)
        centered_text(d, 4+(ICON_W+20)//2, 4+ICON_H//2, "! Notif", FONT_SMALL, fill=(255,255,255))

    push(img)

    # Hit-boxes returned as dict
    return {
        "wallpaper": (4, HEIGHT-ICON_H-4, 4+ICON_W, HEIGHT-4),
        "timer":     (WIDTH-ICON_W-4, HEIGHT-ICON_H-4, WIDTH-4, HEIGHT-4),
        "notif":     (4, 4, 4+ICON_W+20, 4+ICON_H) if notification else None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def draw_notification(message, detail=""):
    img, d = blank_canvas()

    # Alert panel
    px, py = 20, 60
    pw, ph = WIDTH-40, HEIGHT-120
    rounded_rect(d, px, py, px+pw, py+ph, radius=14, fill=C_PANEL, outline=C_ACCENT2)

    # Icon ring
    d.ellipse([px+pw//2-28, py+10, px+pw//2+28, py+66], outline=C_ACCENT2, width=3)
    centered_text(d, px+pw//2, py+38, "!", FONT_XLARGE, fill=C_ACCENT2)

    # Message
    # Simple word-wrap
    words = message.split()
    lines, line = [], ""
    max_w = pw - 20
    for w in words:
        test = (line + " " + w).strip()
        tw, _ = text_size(d, test, FONT_MED)
        if tw <= max_w:
            line = test
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)

    ty = py + 80
    for ln in lines:
        centered_text(d, px+pw//2, ty, ln, FONT_MED, fill=C_TEXT)
        ty += 26

    if detail:
        centered_text(d, px+pw//2, ty+10, detail, FONT_SMALL, fill=C_SUBTEXT)

    # Acknowledge button
    bx0, by0 = WIDTH//2-60, HEIGHT-52
    bx1, by1 = WIDTH//2+60, HEIGHT-12
    rounded_rect(d, bx0, by0, bx1, by1, fill=C_OK, outline=None)
    centered_text(d, WIDTH//2, (by0+by1)//2, "Acknowledge", FONT_MED, fill=(0,0,0))

    draw_realtime_clock(d)
    push(img)
    return {"ack": (bx0, by0, bx1, by1)}

# ─────────────────────────────────────────────────────────────────────────────
# TIMER SETUP SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def draw_timer_setup(hours, minutes, seconds, editing_field):
    """editing_field: 0=hours, 1=minutes, 2=seconds"""
    img, d = blank_canvas()
    draw_realtime_clock(d)

    centered_text(d, WIDTH//2, 28, "Set Timer", FONT_LARGE, fill=C_ACCENT)

    labels = ["HH", "MM", "SS"]
    values = [hours, minutes, seconds]
    field_x = [60, WIDTH//2, WIDTH-60]
    field_y = HEIGHT//2

    SEG_W, SEG_H = 70, 56

    for i in range(3):
        cx = field_x[i]
        is_active = (i == editing_field)
        col = C_ACCENT if is_active else C_BTN
        rounded_rect(d, cx-SEG_W//2, field_y-SEG_H//2,
                        cx+SEG_W//2, field_y+SEG_H//2,
                        fill=col, outline=C_ACCENT if is_active else C_SUBTEXT)
        centered_text(d, cx, field_y - 10, "{:02d}".format(values[i]), FONT_XLARGE,
                      fill=(0,0,0) if is_active else C_TEXT)
        centered_text(d, cx, field_y+SEG_H//2+12, labels[i], FONT_SMALL, fill=C_SUBTEXT)

        # Up / Down arrows
        ax0, ax1 = cx-20, cx+20
        # Up
        rounded_rect(d, ax0, field_y-SEG_H//2-36, ax1, field_y-SEG_H//2-6, fill=C_BTN_HI)
        centered_text(d, cx, field_y-SEG_H//2-21, "▲", FONT_MED, fill=C_TEXT)
        # Down
        rounded_rect(d, ax0, field_y+SEG_H//2+6, ax1, field_y+SEG_H//2+36, fill=C_BTN_HI)
        centered_text(d, cx, field_y+SEG_H//2+21, "▼", FONT_MED, fill=C_TEXT)

        # Colons between fields
        if i < 2:
            mid_x = (field_x[i] + field_x[i+1]) // 2
            centered_text(d, mid_x, field_y-6, ":", FONT_XLARGE, fill=C_SUBTEXT)

    # Start / Cancel buttons
    rounded_rect(d, 20, HEIGHT-44, WIDTH//2-10, HEIGHT-6, fill=C_OK)
    centered_text(d, WIDTH//4+5, HEIGHT-25, "▶ Start", FONT_MED, fill=(0,0,0))

    rounded_rect(d, WIDTH//2+10, HEIGHT-44, WIDTH-20, HEIGHT-6, fill=C_ACCENT2)
    centered_text(d, 3*WIDTH//4-5, HEIGHT-25, "✕ Cancel", FONT_MED, fill=(255,255,255))

    push(img)
    SEG_HALF = SEG_W//2
    hits = {}
    for i in range(3):
        cx = field_x[i]
        ax0, ax1 = cx-20, cx+20
        fh = SEG_H//2
        hits[f"up_{i}"]   = (ax0, field_y-fh-36, ax1, field_y-fh-6)
        hits[f"down_{i}"] = (ax0, field_y+fh+6,  ax1, field_y+fh+36)
        hits[f"field_{i}"]= (cx-SEG_HALF, field_y-fh, cx+SEG_HALF, field_y+fh)
    hits["start"]  = (20, HEIGHT-44, WIDTH//2-10, HEIGHT-6)
    hits["cancel"] = (WIDTH//2+10, HEIGHT-44, WIDTH-20, HEIGHT-6)
    return hits

# ─────────────────────────────────────────────────────────────────────────────
# TIMER RUNNING SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def draw_timer_run(remaining_secs, total_secs):
    img, d = blank_canvas()
    draw_realtime_clock(d)

    centered_text(d, WIDTH//2, 26, "Timer", FONT_MED, fill=C_SUBTEXT)

    h = remaining_secs // 3600
    m = (remaining_secs % 3600) // 60
    s = remaining_secs % 60
    timer_str = "{:02d}:{:02d}:{:02d}".format(h, m, s)
    centered_text(d, WIDTH//2, HEIGHT//2 - 20, timer_str, FONT_XLARGE, fill=C_ACCENT)

    # Progress arc (simple bar)
    progress = 1.0 - (remaining_secs / max(total_secs, 1))
    bar_w = WIDTH - 60
    bar_x = 30
    bar_y = HEIGHT//2 + 40
    BAR_H = 14
    # Background
    rounded_rect(d, bar_x, bar_y, bar_x+bar_w, bar_y+BAR_H, fill=C_BTN)
    # Fill
    fill_w = int(bar_w * progress)
    if fill_w > 4:
        rounded_rect(d, bar_x, bar_y, bar_x+fill_w, bar_y+BAR_H, fill=C_OK)

    # End timer button
    bx0, by0 = WIDTH//2-60, HEIGHT-44
    bx1, by1 = WIDTH//2+60, HEIGHT-6
    rounded_rect(d, bx0, by0, bx1, by1, fill=C_ACCENT2)
    centered_text(d, WIDTH//2, (by0+by1)//2, "⏹ End", FONT_MED, fill=(255,255,255))

    push(img)
    return {"end": (bx0, by0, bx1, by1)}

# ─────────────────────────────────────────────────────────────────────────────
# TIMER DONE SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def draw_timer_done():
    img, d = blank_canvas()
    draw_realtime_clock(d)

    # Pulsing feel – draw a glowing ring
    cx, cy = WIDTH//2, HEIGHT//2 - 20
    for r, alpha in [(70, 40), (55, 80), (40, 160)]:
        ring = Image.new("RGBA", (WIDTH, HEIGHT), (0,0,0,0))
        rd   = ImageDraw.Draw(ring)
        col  = (*C_ACCENT2, alpha)
        rd.ellipse([cx-r, cy-r, cx+r, cy+r], outline=col, width=4)
        img = Image.alpha_composite(img.convert("RGBA"), ring).convert("RGB")
    d = ImageDraw.Draw(img)

    centered_text(d, WIDTH//2, cy, "✓", FONT_XLARGE, fill=C_OK)
    centered_text(d, WIDTH//2, HEIGHT//2 + 50, "Time's Up!", FONT_LARGE, fill=C_ACCENT2)

    bx0, by0 = WIDTH//2-60, HEIGHT-44
    bx1, by1 = WIDTH//2+60, HEIGHT-6
    rounded_rect(d, bx0, by0, bx1, by1, fill=C_OK)
    centered_text(d, WIDTH//2, (by0+by1)//2, "Acknowledge", FONT_MED, fill=(0,0,0))

    push(img)
    return {"ack": (bx0, by0, bx1, by1)}

# ─────────────────────────────────────────────────────────────────────────────
# WALLPAPER PICKER SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def draw_wallpaper_picker(current_idx):
    img, d = blank_canvas()
    draw_realtime_clock(d)

    centered_text(d, WIDTH//2, 28, "Choose Wallpaper", FONT_MED, fill=C_ACCENT)

    thumbs_per_row = 3
    thumb_w = (WIDTH - 40) // thumbs_per_row
    thumb_h = 80
    margin  = 10
    start_y = 50
    hits    = {}

    for i, wp in enumerate(WALLPAPERS):
        col  = i % thumbs_per_row
        row  = i // thumbs_per_row
        x0   = 10 + col * (thumb_w + margin//2)
        y0   = start_y + row * (thumb_h + margin)
        x1   = x0 + thumb_w
        y1   = y0 + thumb_h

        try:
            thumb = load_wallpaper(wp).resize((thumb_w, thumb_h), Image.BICUBIC)
            img.paste(thumb, (x0, y0))
        except Exception:
            rounded_rect(d, x0, y0, x1, y1, fill=C_BTN)

        outline = C_ACCENT if i == current_idx else C_SUBTEXT
        d.rectangle([x0, y0, x1, y1], outline=outline, width=3 if i == current_idx else 1)
        hits[f"wp_{i}"] = (x0, y0, x1, y1)

    # Back button
    bx0, by0 = WIDTH//2-50, HEIGHT-44
    bx1, by1 = WIDTH//2+50, HEIGHT-6
    rounded_rect(d, bx0, by0, bx1, by1, fill=C_BTN, outline=C_ACCENT)
    centered_text(d, WIDTH//2, (by0+by1)//2, "← Back", FONT_MED, fill=C_ACCENT)
    hits["back"] = (bx0, by0, bx1, by1)

    push(img)
    return hits

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    state           = STATE_HOME
    wallpaper_idx   = 0
    hits            = {}

    # Notification queue
    pending_notif   = {"msg": "System Ready!", "detail": "Touch to acknowledge"}

    # Timer state
    timer_h, timer_m, timer_s = 0, 1, 0   # default 1 minute
    editing_field  = 1                     # start editing minutes
    timer_end_time = None
    timer_total    = 0

    last_render    = 0
    RENDER_RATE    = 0.2   # seconds between full redraws

    # Touch debounce
    last_touch_time = 0
    DEBOUNCE        = 0.25

    # ── initial render ───────────────────────────────────────────────────────
    if pending_notif:
        state = STATE_NOTIFICATION
        hits  = draw_notification(pending_notif["msg"], pending_notif.get("detail",""))
    else:
        hits  = draw_home(wallpaper_idx)

    while True:
        now = time.monotonic()
        touch = get_touch()

        # ── handle touch events ──────────────────────────────────────────────
        if touch and (now - last_touch_time) > DEBOUNCE:
            tx, ty = touch
            last_touch_time = now

            # ---- HOME -------------------------------------------------------
            if state == STATE_HOME:
                if hits.get("wallpaper") and touch_in(tx, ty, *hits["wallpaper"]):
                    state = STATE_WALLPAPER
                    hits  = draw_wallpaper_picker(wallpaper_idx)
                elif hits.get("timer") and touch_in(tx, ty, *hits["timer"]):
                    state = STATE_TIMER_SETUP
                    hits  = draw_timer_setup(timer_h, timer_m, timer_s, editing_field)
                elif hits.get("notif") and hits["notif"] and touch_in(tx, ty, *hits["notif"]):
                    state = STATE_NOTIFICATION
                    hits  = draw_notification(pending_notif["msg"], pending_notif.get("detail",""))

            # ---- NOTIFICATION -----------------------------------------------
            elif state == STATE_NOTIFICATION:
                if touch_in(tx, ty, *hits["ack"]):
                    pending_notif = None
                    state = STATE_HOME
                    hits  = draw_home(wallpaper_idx)

            # ---- TIMER SETUP ------------------------------------------------
            elif state == STATE_TIMER_SETUP:
                if touch_in(tx, ty, *hits["start"]):
                    total = timer_h*3600 + timer_m*60 + timer_s
                    if total > 0:
                        timer_total    = total
                        timer_end_time = time.monotonic() + total
                        state = STATE_TIMER_RUN
                        hits  = draw_timer_run(total, total)
                elif touch_in(tx, ty, *hits["cancel"]):
                    state = STATE_HOME
                    hits  = draw_home(wallpaper_idx, pending_notif)
                else:
                    # Field select
                    for i in range(3):
                        if touch_in(tx, ty, *hits[f"field_{i}"]):
                            editing_field = i
                        if touch_in(tx, ty, *hits[f"up_{i}"]):
                            if   i == 0: timer_h = (timer_h + 1) % 24
                            elif i == 1: timer_m = (timer_m + 1) % 60
                            else:        timer_s = (timer_s + 1) % 60
                        if touch_in(tx, ty, *hits[f"down_{i}"]):
                            if   i == 0: timer_h = (timer_h - 1) % 24
                            elif i == 1: timer_m = (timer_m - 1) % 60
                            else:        timer_s = (timer_s - 1) % 60
                    hits = draw_timer_setup(timer_h, timer_m, timer_s, editing_field)

            # ---- TIMER RUNNING ----------------------------------------------
            elif state == STATE_TIMER_RUN:
                if touch_in(tx, ty, *hits["end"]):
                    timer_end_time = None
                    state = STATE_HOME
                    hits  = draw_home(wallpaper_idx, pending_notif)

            # ---- TIMER DONE -------------------------------------------------
            elif state == STATE_TIMER_DONE:
                if touch_in(tx, ty, *hits["ack"]):
                    state = STATE_HOME
                    hits  = draw_home(wallpaper_idx, pending_notif)

            # ---- WALLPAPER PICKER -------------------------------------------
            elif state == STATE_WALLPAPER:
                if touch_in(tx, ty, *hits["back"]):
                    state = STATE_HOME
                    hits  = draw_home(wallpaper_idx, pending_notif)
                else:
                    for i in range(len(WALLPAPERS)):
                        key = f"wp_{i}"
                        if key in hits and touch_in(tx, ty, *hits[key]):
                            wallpaper_idx = i
                    hits = draw_wallpaper_picker(wallpaper_idx)

        # ── periodic redraws (clock + timer tick) ────────────────────────────
        if now - last_render > RENDER_RATE:
            last_render = now

            if state == STATE_HOME:
                hits = draw_home(wallpaper_idx, pending_notif)

            elif state == STATE_TIMER_RUN:
                remaining = max(0, int(timer_end_time - time.monotonic()))
                if remaining == 0:
                    state = STATE_TIMER_DONE
                    hits  = draw_timer_done()
                else:
                    hits = draw_timer_run(remaining, timer_total)

            elif state == STATE_TIMER_DONE:
                hits = draw_timer_done()

            elif state == STATE_NOTIFICATION:
                # just refresh the clock
                pass  # clock is embedded in draw; redraw every cycle would flicker
                      # notification is static so skip continuous redraw unless needed

        time.sleep(0.05)

if __name__ == "__main__":
    main()