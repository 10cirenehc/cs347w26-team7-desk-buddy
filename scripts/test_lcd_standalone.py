#!/usr/bin/env python3
"""Standalone LCD test — no coaster, no GPIO.BOARD conflict."""

import sys
import time
sys.path.insert(0, ".")

from PIL import Image, ImageDraw, ImageFont

def main():
    print("=== LCD Standalone Test ===")

    # 1. Check if Blinka / LCD libs are available
    try:
        import board
        import digitalio
        import busio
        import adafruit_rgb_display.ili9341 as ili9341
        import adafruit_focaltouch
        print("[OK] LCD libraries available")
    except ImportError as e:
        print(f"[FAIL] Missing library: {e}")
        return

    # 2. Check current GPIO mode (should be None at start)
    try:
        import Jetson.GPIO as GPIO
        mode = GPIO.getmode()
        print(f"[INFO] GPIO mode before LCD init: {mode}")
    except ImportError:
        print("[INFO] Jetson.GPIO not available (not on Jetson?)")

    # 3. Initialize LCD hardware
    try:
        cs_pin = digitalio.DigitalInOut(board.CE0)
        dc_pin = digitalio.DigitalInOut(board.D25)
        reset_pin = digitalio.DigitalInOut(board.D24)
        spi = board.SPI()
        disp = ili9341.ILI9341(
            spi, rotation=90,
            cs=cs_pin, dc=dc_pin, rst=reset_pin, baudrate=24000000,
        )
        print(f"[OK] LCD initialized: {disp.width}x{disp.height}")
    except Exception as e:
        print(f"[FAIL] LCD init failed: {e}")
        import traceback; traceback.print_exc()
        return

    # 4. Check GPIO mode after LCD init (Blinka sets this)
    try:
        mode = GPIO.getmode()
        print(f"[INFO] GPIO mode after LCD init: {mode}")
        if mode is not None:
            print(f"       BOARD={GPIO.BOARD}, BCM={GPIO.BCM}, TEGRA_SOC={getattr(GPIO, 'TEGRA_SOC', '?')}")
    except Exception:
        pass

    # 5. Draw test pattern
    try:
        img = Image.new("RGB", (320, 240), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 310, 230], outline=(0, 255, 0), width=3)
        draw.text((80, 100), "LCD TEST OK", fill=(255, 255, 255))
        draw.text((80, 130), time.strftime("%H:%M:%S"), fill=(0, 255, 0))
        disp.image(img)
        print("[OK] Test image displayed")
    except Exception as e:
        print(f"[FAIL] Display render failed: {e}")
        import traceback; traceback.print_exc()
        return

    # 6. Test touch
    try:
        i2c = busio.I2C(board.SCL_1, board.SDA_1)
        ft = adafruit_focaltouch.Adafruit_FocalTouch(i2c, debug=False)
        print("[OK] Touch controller initialized")
        print("Touch the screen within 10 seconds...")
        end = time.time() + 10
        while time.time() < end:
            if ft.touched:
                touches = ft.touches
                if touches:
                    t = touches[0]
                    print(f"  Touch at raw=({t['x']}, {t['y']})")
            time.sleep(0.1)
    except Exception as e:
        print(f"[WARN] Touch test failed: {e}")

    print("\n=== LCD test complete ===")


if __name__ == "__main__":
    main()
