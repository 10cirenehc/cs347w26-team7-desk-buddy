
'''
Smart Coaster v1.0
March 04, 2026
'''

import os
import time
import json
import Jetson.GPIO as GPIO
from hx711 import HX711 # load cell library
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional

#
# CONSTANTS
#
PROFILE_FILE = "coaster_profiles.json"
REMINDER_FREQ_SECONDS = (30 * 60) # every 30 minutes
SIP_THRESHOLD_GRAMS = 10 # min for what counts as a sip
EMPTY_CUP_GRAMS = 5 # empty cup or no cup on coaster

#
# SCALE CLASS
#
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

DT_PIN = 29
SCK_PIN = 31

class Scale:
      def __init__(self, load_cell: HX711):
            load_cell

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
# JSON
#
def load_profile() -> dict[str, Profile]:
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

def save_profile(profiles: dict[str, Profile]) -> None:
      bulk_list = {}
      for name, profile in profiles.items():
            entry = asdict(profile)
            bulk_list[name] = entry
      with open(PROFILE_FILE, "w") as file:
            json.dump(bulk_list, file, indent = 2)
      print(f"Written to {PROFILE_FILE}")

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
                        self.profile.last_drink_time = datetime.now().isoformat()
                  self.last_reminder = time.time() # update reminder timer in profile
                  return sip_amount
            return None
      
      # Mainly for display
      def get_status(self) -> str:
            goal = self.profile.daily_goal_ml if self.profile else None
            msg = [f"\tTotal intake:\t{self.intake_ml:..0f} mL"]
            if goal:
                  percent = min(100, (self.intake_ml / goal * 100))
                  msg.append(f"\tDaily goal:\t{goal:.0f} mL ({percent:.0f}% reached)")
            if self.cup:
                  msg.append(f"\tCurrent cup:\t{self.cup.name} (tare: {self.cup.cup_weight_grams}g)")
            return "\n".join(msg)
      
#
# FREE MODE
#
def free_mode():
      print("ENTERING FREE MODE")
      # TO FIX SHOULD BE READING FROM LOAD CELL
      cup_wgt = input("Enter empty cup weight in grams (or 0 to skip): ")
      try:
            tare = float(cup_wgt)
      except ValueError:
            tare = 0.0
      
      cup = Cup(name="Unknown Cup", cup_weight_grams=tare) if tare > 0 else None
      coaster = SmartCoaster(profile=None)
      coaster.cup = cup

      # Free Mode Main Loop
      print("Free mode running.")
      while True:
            weight =  # load cell read
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
def select_profile(profiles: dict[str, Profile]) -> Profile:
      if profiles:
            print
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
      if profile.last_cup:
            print(f"\Most recent cup: '{profile.recent_cup}' (tare: {profile.recent_cup.cup_weight_grams}g)")
            use_last = input("\tWould you like to use this cup again? [y/n]").strip().lower()
            if use_last != "n":
                  return profile.recent_cup
      name = input("Enter cup name: ").strip() or "Default Cup"
      tare = input("Enter cup tare weight in grams: ").strip()
      cup = Cup(name=name, tare_weight_grams=float(tare) if tare else 0.0)
      profile.recent_cup = cup # save for next time
      return cup


def profile_mode(profiles: dict[str, Profile]):
      print("ENTERING PROFILE MODE")
      profile = select_profile(profiles)
      cup = select_cup(profile)

#
# MAIN FUNCTION
#
def main():
      profiles = load_profile()
      myLoad = HX711(SCK_PIN, DT_PIN, 128)

      mode = input("Select mode: ").strip() # removing whitespace

      if mode == "":
            profile_mode(profiles)
      elif mode == "":
            free_mode()
      else:
            print("ERROR:\tinvalid mode choice.")
            print("Exiting program.")

if __name__ == "__main__":
      main()