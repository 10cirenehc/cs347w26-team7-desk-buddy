# Example Project -- using to help calibrate...
import time

import board
import digitalio

# Using Adafruit CircuitPython HX711 library
from adafruit_hx711.hx711 import HX711
from adafruit_hx711.analog_in import AnalogIn

# Data Pin
data = digitalio.DigitalInOut(board.D2)
data.direction = digitalio.Direction.INPUT
# Clock Pin
clock = digitalio.DigitalInOut(board.D3)
clock.direction = digitalio.Direction.OUTPUT

hx711 = HX711(data, clock)
# We are using Channel A
channel_a = AnalogIn(hx711, HX711.CHAN_A_GAIN_128)

while True:
    print(f"Reading: {channel_a.value}")
    time.sleep(1)
