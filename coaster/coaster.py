'''
Actual Coaster Implementation
Katie Jiang
'''

# imports
import time
import board
import digitalio

# Using Adafruit CircuitPython HX711 library
from adafruit_hx711.hx711 import HX711
from adafruit_hx711.analog_in import AnalogIn

# Cup Class
from cup import Cup

# Data Pin
data = digitalio.DigitalInOut(board.D2)
data.direction = digitalio.Direction.INPUT
# Clock Pin
clock = digitalio.DigitalInOut(board.D3)
clock.direction = digitalio.Direction.OUTPUT

hx711 = HX711(data, clock)
# We are using Channel A
channel_a = AnalogIn(hx711, HX711.CHAN_A_GAIN_128)

# User Prompting
profile_mode = True # assume true for now
print("Place your empty cup on the coaster.\n")
time.sleep(1)
print("Fill your cup to capacity, then place it back on the coaster.")
time.sleep(1)
my_cup_name = input("Please name the cup:\n")

# NEED TO STORE CUP PROFILE SOMEWHERE -- STORE ONLY THE LAST-USED CUP?
if (profile_mode):
    my_cup = Cup(my_cup_name, cupweight, weightoffullcup)
    
while True:
    print(f"Reading: {channel_a.value}")
    time.sleep(1)