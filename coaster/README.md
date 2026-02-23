## Smart Coaster Feature
## Cup Class `cup.py`
### Attributes
These are passed into the __init__:
- name
- cup weight
- full cup weight
### Getter Functions
get_cup_weight(): returns weight of solely the cup
get_total_water(): returns

get_percent_water(current_weight):
            curr_water_w = curr_w - self.__weight
            return (curr_water_w / self.__full_water)

## Pin Connections (HX711 -> Jetson)
I have physically connected VDD and VCC on the HX711 board.
- VDD -> 3.3V
- VCC -> 17 (3.3V)
- CLK -> 31 (Input)
- Data -> 29 (Output)

TO DO: Save most recently used cup in CSV file. In the future, can implement multiple cups but for MVP, only use most recently used?