#!/usr/bin/env python3
"""Test HX711 load cell with TEGRA_SOC GPIO mode (simulating Blinka/LCD context).

Run on Jetson AGX Orin:
    cd /path/to/desk-buddy
    python3 scripts/test_coaster_tegra_soc.py
"""
import sys
import time
import signal

sys.path.insert(0, "coaster")

import Jetson.GPIO as GPIO
from hx711 import HX711

GPIO.setwarnings(False)
GPIO.setmode(GPIO.TEGRA_SOC)

# Pin 31 (BOARD) = CAN0_DOUT -> SCK
# Pin 29 (BOARD) = CAN0_DIN  -> DT (DOUT on HX711)

# Attempt 1: Full TEGRA_SOC names from gpio_pin_data.py
PIN_SETS = [
    ("GP17_CAN0_DOUT", "GP18_CAN0_DIN", "TEGRA_SOC full names"),
    ("CAN0_DOUT", "CAN0_DIN", "CVM/alternate names"),
]


def timeout_handler(signum, frame):
    raise TimeoutError("HX711 read timed out - DOUT never went low")


def test_pins(sck_pin, dt_pin, label):
    print(f"\n--- Testing: {label} ---")
    print(f"  GPIO mode: {GPIO.getmode()}")
    print(f"  SCK={sck_pin}, DT={dt_pin}")

    try:
        lc = HX711(sck_pin, dt_pin, 128)
        print("  HX711 initialized, attempting read (5s timeout)...")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        val = lc.read()
        signal.alarm(0)
        print(f"  SUCCESS: read value = {val}")
        return True
    except TimeoutError as e:
        signal.alarm(0)
        print(f"  FAIL: {e}")
        print("  The pin names may not be routing to the correct physical pins.")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    finally:
        try:
            GPIO.cleanup()
            # Re-set mode for next attempt
            GPIO.setmode(GPIO.TEGRA_SOC)
        except Exception:
            pass


def main():
    print("=" * 60)
    print("HX711 Load Cell - TEGRA_SOC GPIO Mode Test")
    print("=" * 60)

    for sck_pin, dt_pin, label in PIN_SETS:
        if test_pins(sck_pin, dt_pin, label):
            print(f"\n>>> Pin set '{label}' works! Use these in smart_coaster.py")
            break
    else:
        print("\n>>> All TEGRA_SOC pin names failed.")
        print(">>> Debug: check physical pin mapping with:")
        print(">>>   cat /sys/kernel/debug/gpio")
        print(">>>   sudo cat /sys/kernel/debug/pinctrl/2200000.gpio/pins")
        print(">>> Fallback: consider subprocess isolation (BOARD mode in separate process)")

    GPIO.cleanup()


if __name__ == "__main__":
    main()
