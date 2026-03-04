# Smart Coaster

# Software/API

## HX711 Class
Found in `hx711.py`

## SmartCoaster Class
Found in `coaster.py`

## Dataclasses
Found in `coaster.py`
### Calibration
### Cup
### Profile

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

## Coaster 3D Print
Made by Janet Bai:

[Link](https://cad.onshape.com/documents/2e91cbeae698b6ab89716089/w/e1222bfd9f4ff7c4da097b5e/e/e275416058a4f195d6499203?renderMode=0&uiState=69a8888a4e89299f92843471)
