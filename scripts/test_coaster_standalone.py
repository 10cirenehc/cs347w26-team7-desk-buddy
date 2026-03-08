#!/usr/bin/env python3
"""Standalone coaster/HX711 test — no LCD, no Blinka conflict."""

import sys
import time
sys.path.insert(0, ".")
sys.path.insert(0, "coaster")

def main():
    print("=== Coaster Standalone Test ===")

    # 1. Check Jetson.GPIO
    try:
        import Jetson.GPIO as GPIO
        print("[OK] Jetson.GPIO available")
    except ImportError:
        print("[FAIL] Jetson.GPIO not available")
        return

    # 2. Check current GPIO mode
    mode = GPIO.getmode()
    print(f"[INFO] GPIO mode before setup: {mode}")

    # 3. Set BOARD mode and init HX711
    try:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        print(f"[OK] GPIO mode set to BOARD ({GPIO.BOARD})")
    except Exception as e:
        print(f"[FAIL] Cannot set GPIO.BOARD: {e}")
        return

    DT_PIN = 29
    SCK_PIN = 31

    try:
        from hx711 import HX711
        load_cell = HX711(SCK_PIN, DT_PIN, 128)
        print(f"[OK] HX711 initialized (SCK={SCK_PIN}, DT={DT_PIN})")
    except Exception as e:
        print(f"[FAIL] HX711 init failed: {e}")
        import traceback; traceback.print_exc()
        GPIO.cleanup()
        return

    # 4. Read raw values
    print("\nReading 5 samples (raw)...")
    for i in range(5):
        try:
            raw = load_cell.read()
            print(f"  Sample {i+1}: {raw}")
        except Exception as e:
            print(f"  Sample {i+1}: ERROR - {e}")
        time.sleep(0.5)

    # 5. Read average
    print("\nReading average (10 samples)...")
    try:
        avg = load_cell.read_average(10)
        print(f"  Average: {avg}")
    except Exception as e:
        print(f"  Average: ERROR - {e}")

    # 6. Test with calibration
    import json
    cal_path = "coaster/calibration.json"
    try:
        with open(cal_path) as f:
            cal = json.load(f)
        zero = cal["raw_zero_value"]
        scale = cal["scale_factor"]
        raw = load_cell.read_average(10)
        grams = (raw - zero) / scale if scale != 0 else 0
        print(f"\n[INFO] Calibration: zero={zero}, scale={scale}")
        print(f"[INFO] Weight reading: {grams:.1f} g")
    except FileNotFoundError:
        print(f"\n[WARN] No calibration file at {cal_path}")
    except Exception as e:
        print(f"\n[WARN] Calibration test failed: {e}")

    # 7. Check GPIO mode at end
    mode = GPIO.getmode()
    print(f"\n[INFO] GPIO mode at end: {mode}")

    GPIO.cleanup()
    print("\n=== Coaster test complete ===")


if __name__ == "__main__":
    main()
