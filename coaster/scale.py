import Jetson.GPIO as GPIO
import time
from hx711 import HX711

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

DT_PIN = 29
SCK_PIN = 31

myLoad = HX711(SCK_PIN, DT_PIN, 128)

while True:
      # print("Clock: ", GPIO.input(DT_PIN))
      print("Read: ", myLoad.read())
      time.sleep(1)