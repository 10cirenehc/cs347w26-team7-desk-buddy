# Smart Coaster

# Software/API

## File Descriptions

`coaster.py`: Smart Coaster System

`calibration.json`: Stores the calibration settings for the load cell scale.

`coaster_profiles.json`: Stores the profiles for Profile Mode.

`hx711.py`: Driver for the HX711 amplifier used with the load cell.

`calibrate.py`: Most simple program to get a read from the load cell. Used for testing components, not for the Smart Coaster.

# Hardware

## Components
- 10kg Load Cell
- SparkFun HX711 Load Cell Amplifier
- Jetson AGX Orin

## Pin Connections (HX711 -> Jetson AGX Orin)
I already soldered VCC and VDD together on the HX711 board, so only one of them needs to be connected (saves a pin on the Jetson).
- VCC or VDD -> 17 (3.3V)
- GND -> 39
- CLK (Blue) -> 31 (Input)
- Data (Purple) -> 29 (Output)