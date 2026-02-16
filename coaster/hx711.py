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
            GPIO.setup(self.__dout, GPIO.IN)

            self.set_gain(gain)
      
      def is_ready(self):
            # When DOUT goes low, data is ready for retrieval
            return (GPIO.input(self.__dout) == 0)
      
      def set_gain(self, gain):
            if (gain == 128):
                  self.__gpulse = 1
            elif (gain == 64):
                  self.__gpulse = 3
            elif (gain == 32): 
                  self.__gpulse = 2
            else:
                  self.__gpulse = 0
                  print("Error: Invalid HX711 Gain")
            # match gain:
            #       case 128: # Channel A, gain factor 128
            #             self.__gpulse = 1
            #       case 64: # Ch A, gain factor 64
            #             self.__gpulse = 3
            #       case 32: # Ch B
            #             self.__gpulse = 2
            #       case _: # Error
            #             self.__gpulse = 0
            #             return "Error: Invalid HX711 Gain"
      
      def read(self):
            value = 0
            # Wait for data to be ready
            while (self.is_ready != True):
                  pass

            # Apply 25~27 positive clock pulses at PD_SCK
            # to shift out data from DOUT (24 bits total)
            for _ in range(24):
                  GPIO.output(self.__pd_sck, True)
                  # MSB bit shifted out first
                  value = (value << 1) | GPIO.input(self.__dout)
                  GPIO.output(self.__pd_sck, False)
            
            # 25th pulse pulls DOUT back to high
            # Additional pulses control input and gain            
            # Good practice would be to disable interrupts here!
            # Set gain accordingly for next read:
            for _ in range(self.__gpulse):
                  GPIO.output(self.__pd_sck, True)
                  time.sleep(0.01) # these delays may not be necessary...
                  GPIO.output(self.__pd_sck, False)
                  time.sleep(0.01)
            # And re-enable them here!
            
            # Signed 24-bit
            value = value ^ 0x800000
            return value