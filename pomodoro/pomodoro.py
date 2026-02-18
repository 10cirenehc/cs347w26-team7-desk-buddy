import time
# do we want to count seconds ourselves?
# or do we want to grab the real time and do subtraction?

class Pomodoro:
      def __init__(self, total_time):
            self.length = total_time
            self.elapsed = 0
      
      def restart_timer(self):
            self.elapsed = 0