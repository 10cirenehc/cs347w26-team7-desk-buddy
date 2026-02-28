class Cup:
      def __init__(self, name, cup_weight, full_cup_weight):
            self.name = name
            self.__weight = cup_weight
            self.__full_water = full_cup_weight - cup_weight
      
      def get_cup_weight(self):
            return self.__weight
      
      def get_total_water(self):
            return self.__full_water
      
      def get_percent_water(self, curr_w):
            curr_water_w = curr_w - self.__weight
            return (curr_water_w / self.__full_water)