from datetime import datetime
import time

# Variables
last_pickup = datetime.now()
# profile_mode = 1 for profile mode, else 0 for free mode
profile_mode = 0 

# Helper Functions
def minutes_passed_since(hour, minute):
      current_time = datetime.now()
      if (current_time.hour == hour):
            return (current_time.minute - minute)
      else:
            return ((60*(current_time.hour - hour)) + current_time.minute) - minute

# Free mode
print(datetime.now())

# Loop
while (True):
      if (minutes_passed_since(last_pickup.hour, last_pickup.minute) > 30):
            print("Reminder: Drink Water!")