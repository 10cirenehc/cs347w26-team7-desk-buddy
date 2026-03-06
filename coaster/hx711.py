'''
Driver for HX711 on Jetson GPIO
Based on the Reference Driver (C)
in the Datasheet
'''

import Jetson.GPIO as GPIO
import time

class HX711:
      def __init__(self, sck_pin, dout_pin, gain):
            self.__pd_sck = sck_pin
            self.__dout = dout_pin
            self.__gpulse = 0

            GPIO.setup(self.__pd_sck, GPIO.OUT)
            GPIO.output(self.__pd_sck, False)
            GPIO.setup(self.__dout, GPIO.IN)

            self.set_gain(gain)
            self.reset()
            self.read() # dummy read
      
      def reset(self) -> None:
        # Power down then power back up
        GPIO.output(self.__pd_sck, True)
        time.sleep(0.0001)
        GPIO.output(self.__pd_sck, False)
        time.sleep(0.4)

      def is_ready(self) -> bool:
            # When DOUT goes low, data is ready for retrieval
            return (not GPIO.input(self.__dout))
      
      def set_gain(self, gain) -> None:
            if (gain == 128): # Channel A, gain factor 128
                  self.__gpulse = 1
            elif (gain == 64): # Ch A, gain factor 64
                  self.__gpulse = 3
            elif (gain == 32): # Ch B
                  self.__gpulse = 2
            else: # Error
                  self.__gpulse = 0
                  print("Error: Invalid HX711 Gain")
      
      def read(self):
            value = 0
            # Wait for data to be ready
            while not self.is_ready():
                  print("Load cell is not ready!")
                  pass

            # Apply 25~27 positive clock pulses at PD_SCK
            # to shift out data from DOUT (24 bits total)
            for _ in range(24):
                  GPIO.output(self.__pd_sck, True)
                  bit = GPIO.input(self.__dout)
                  # MSB bit shifted out first
                  GPIO.output(self.__pd_sck, False)
                  value = (value << 1) | bit
            
            # 25th pulse pulls DOUT back to high
            # Additional pulses control input and gain            
            # Good practice would be to disable interrupts here!
            # Set gain accordingly for next read:
            for _ in range(self.__gpulse):
                  GPIO.output(self.__pd_sck, True)
                  GPIO.output(self.__pd_sck, False)
            # And re-enable them here!
            
            # Signed 24-bit
            if value & 0x800000:
                  value -= 1 << 24
            return value
      
      # Average out the analog inputs to get a steady weight
      def read_average(self, samples = 10):
            total = 0
            for _ in range(samples):
                  total += self.read()
            return total / samples