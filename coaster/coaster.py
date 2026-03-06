
'''
Smart Coaster v1.0
March 05, 2026
'''

import os
import time
import json
import Jetson.GPIO as GPIO
from hx711 import HX711 # load cell library
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict

#
# CONSTANTS
#
PROFILE_FILE = "coaster_profiles.json"
CALIBRATION_FILE = "calibration.json"
REMINDER_FREQ_SECONDS = (30 * 60) # every 30 minutes
SIP_THRESHOLD_GRAMS = 10 # min for what counts as a sip
EMPTY_CUP_GRAMS = 5 # empty cup or no cup on coaster
NUM_SAMPLES = 1000

#
# LOAD CELL CONFIG
#
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

DT_PIN = 29
SCK_PIN = 31

#
# DATACLASSES
#
@dataclass
class Cup:
      name: str # name of cup
      cup_weight_grams: float # empty cup weight

@dataclass
class Profile:
      name: str
      daily_goal_ml: float = 2000.0
      reminder_frequency: float = REMINDER_FREQ_SECONDS
      recent_cup: Optional[Cup] = None
      last_sip_time: Optional[str] = None

#
# PROFILES.JSON
#
def load_profile() -> Dict[str, Profile]:
      if not os.path.exists(PROFILE_FILE):
            return {}
      with open(PROFILE_FILE, "r") as file:
            raw = json.load(file)
      profiles = {}
      for name, data in raw.items():
            cup_info = data.pop("recent_cup", None)
            cup = Cup(**cup_info) if cup_info else None # ternary
            profiles[name] = Profile(**data, recent_cup=cup)
      return profiles

def save_profile(profiles: Dict[str, Profile]) -> None:
      bulk_list = {}
      for name, profile in profiles.items():
            entry = asdict(profile)
            bulk_list[name] = entry
      with open(PROFILE_FILE, "w") as file:
            json.dump(bulk_list, file, indent = 2)
      print(f"Written to {PROFILE_FILE}")

#
# CALIBRATION
#
@dataclass
class Calibration:
      raw_zero_value: float
      raw_ref_value: float
      ref_weight_grams: float
      scale_factor: float
      last_calibrated: str

def load_calibration() -> Optional[Calibration]:
      if not os.path.exists(CALIBRATION_FILE):
            return None
      with open(CALIBRATION_FILE, "r") as file:
            data = json.load(file)
      return Calibration(**data)

def save_calibration(cal: Calibration) -> None:
      with open(CALIBRATION_FILE, "w") as file:
            json.dump(asdict(cal), file, indent=2)
      print(f"{CALIBRATION_FILE} updated.")

def calibrate(load_cell: HX711) -> bool:
      print("ENTERING CALIBRATION")
      response = input("Ensure nothing is on the coaster. Enter 'Y' to continue: ").strip().lower()
      if response != "y":
            print("ERROR:\tfailed to calibrate.")
            print("Exiting calibration.")
            return False
      scale_zero = load_cell.read_average(NUM_SAMPLES)
      print("Scale zero has been updated.")
      ref_wgt_grams = input("Choose a reference object. Enter the object's weight in grams: ").strip()
      ref_wgt_grams = float(ref_wgt_grams)
      response = input("Place the reference object on the coaster. Enter 'Y' to continue: ").strip().lower()
      if response != "y":
            print("ERROR:\tfailed to calibrate.")
            print("Exiting calibration.")
            return False
      ref_wgt_analog = load_cell.read_average(NUM_SAMPLES)
      print(f"Reading {ref_wgt_grams}g = {ref_wgt_analog}")
      raw_per_gram = (ref_wgt_analog - scale_zero) / ref_wgt_grams
      cal = Calibration(
            raw_zero_value=scale_zero,
            raw_ref_value=ref_wgt_analog,
            ref_weight_grams=ref_wgt_grams,
            scale_factor=raw_per_gram,
            last_calibrated=datetime.now().isoformat()
      )
      save_calibration(cal)
      print(f"Found scale factor: {raw_per_gram:.4f}")
      print("Calibration complete!")
      return True

#
# LOAD CELL -> GRAMS
#
def read_grams(cal: Calibration, load_cell: HX711) -> float:
      raw = load_cell.read_average(NUM_SAMPLES)
      return ((raw-cal.raw_zero_value)/cal.scale_factor)

#
# COASTER CLASS
#
class SmartCoaster:
      def __init__(self, profile: Optional[Profile] = None):
            self.profile = profile
            self.intake_ml = 0.0
            self.last_reminder = time.time()
            self.last_weight = 0.0
            self.cup: Optional[Cup] = profile.recent_cup if profile else None
      
      def poll_reminder(self) -> bool:
            interval = (self.profile.reminder_frequency if self.profile else REMINDER_FREQ_SECONDS)
            if time.time() - self.last_reminder >= interval:
                  self.last_reminder = time.time()
                  return True
            return False # else
      
      def process_sip(self, weight: float) -> Optional[float]:
            tare = self.cup.cup_weight_grams if self.cup else 0.0
            prev_water_wgt = max(0.0, self.last_weight - tare)
            curr_water_wgt = max(0.0, weight - tare)
            sip_amount = prev_water_wgt - curr_water_wgt # Should be positive if water was drank
            # update weight in profile
            self.last_weight = weight
            if sip_amount >= SIP_THRESHOLD_GRAMS:
                  self.intake_ml += sip_amount
                  if self.profile:
                        self.profile.last_sip_time = datetime.now().isoformat()
                  self.last_reminder = time.time() # update reminder timer in profile
                  return sip_amount
            return None
      
      # Mainly for Display
      def get_status(self) -> str:
            goal = self.profile.daily_goal_ml if self.profile else None
            msg = [f"\tTotal intake:\t{self.intake_ml:.0f} mL"]
            if goal:
                  percent = min(100, (self.intake_ml / goal * 100))
                  msg.append(f"\tDaily goal:\t{goal:.0f} mL ({percent:.0f}% reached)")
            if self.cup:
                  msg.append(f"\tCurrent cup:\t{self.cup.name} (tare: {self.cup.cup_weight_grams}g)")
            return "\n".join(msg)

      def get_total_intake(self) -> float:
            return self.intake_ml
      
      def get_percent_goal(self):
            goal = self.profile.daily_goal_ml if self.profile else None
            if goal:
                  percent = min(100, (self.intake_ml / goal * 100))
                  return percent
            return None
      
#
# FREE MODE
#
def free_mode(cal: Calibration, load_cell: HX711):
      print("ENTERING FREE MODE\n")
      print("Place your empty cup on the coaster.")
      input("\tPress Enter when ready.")
      tare = read_grams(cal, load_cell)
      print(f"\tCup tare recorded: {tare:.1f}g")
      cup = Cup(name="Unknown Cup", cup_weight_grams=tare) if tare > 0 else None      
      coaster = SmartCoaster(profile=None)
      coaster.cup = cup

      # Free Mode Main Loop
      print("Free mode running.")
      while True:
            cmd = input("[free mode] > ").strip().lower()
            if cmd == "q": # quit
                  break
            else:
                  weight = read_grams(cal, load_cell) # load cell read
                  sipped = coaster.process_sip(weight)
                  if sipped:
                        print(f"\t~{sipped:.0f} mL sip detected.")
                  else:
                        print("\tNo significant sip detected.")
                  
                  if coaster.poll_reminder():
                        print("Alert: Time to drink some water")

#
# PROFILE MODE
#
def select_profile(profiles: Dict[str, Profile]) -> Profile:
      if profiles:
            print("\nExisting profiles:",",".join(profiles.keys()))
      name = input("Enter new or existing profile name: ").strip()
      if name in profiles:
            print(f"\tLoading profile: {name}")
            return profiles[name]
      else:
            goal = input("\tDaily water goal in mL (default 2000): ").strip()
            profile = Profile(name=name, daily_goal_ml=float(goal) if goal else 2000.0)
            profiles[name] = profile
            print(f"\tCreated new profile: {name}")
            return profile

def select_cup(profile: Profile) -> Cup:
      if profile.recent_cup:
            print(f"Most recent cup: '{profile.recent_cup.name}' (tare: {profile.recent_cup.cup_weight_grams}g)")
            use_last = input("\tWould you like to use this cup again? [y/n]").strip().lower()
            if use_last != "n":
                  return profile.recent_cup
      name = input("Enter cup name: ").strip() or "Default Cup"
      tare = input("Enter cup tare weight in grams: ").strip()
      cup = Cup(name=name, cup_weight_grams=float(tare) if tare else 0.0)
      profile.recent_cup = cup # save for next time
      return cup


def profile_mode(profiles: Dict[str, Profile], cal: Calibration, load_cell: HX711):
      print("ENTERING PROFILE MODE\n")
      profile = select_profile(profiles)
      cup = select_cup(profile)

      coaster = SmartCoaster(profile=profile)
      coaster.cup = cup

      print("Profile mode running.")
      try:
            while True:
                  cmd = input(f"[{profile.name}] > ").strip().lower()
                  if cmd == "q": # quit
                        break
                  else:
                        weight = read_grams(cal, load_cell)
                        sipped = coaster.process_sip(weight)
                        if sipped:
                              print(f"\t~{sipped:.0f} mL sip detected.")
                        else:
                              print("\tNo significant sip detected.")
                        if coaster.poll_reminder():
                              print("Alert: Time to drink some water")
      finally:
            save_profile(profiles)
            print("Profile saved.")

#
# MAIN FUNCTION
#
def main():
      profiles = load_profile()
      my_load = HX711(SCK_PIN, DT_PIN, 128)

      cal = load_calibration()
      if cal is None:
            print("No calibration found. Please calibrate first.")
            success = calibrate(my_load)
            if not success:
                  return
            cal = load_calibration()  # reload file

      print("1. Profile Mode")
      print("2. Free Mode")
      mode = input("Enter mode number [1/2]: ").strip() # removing whitespace
      if mode == "1":
            profile_mode(profiles, cal, my_load)
      elif mode == "2":
            free_mode(cal, my_load)
      elif mode == "00": # secret code for calibration
            calibrate(my_load)
      else:
            print("ERROR:\tinvalid mode choice.")
            print("Exiting program.")

if __name__ == "__main__":
      main()
