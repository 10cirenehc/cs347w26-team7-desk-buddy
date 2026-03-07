# Smart Coaster
Smart Coaster system for Desk Buddy

# Software/API

## HX711 Class
Found in `hx711.py`

## SmartCoaster Class
Found in `coaster.py`

## Dataclasses
Found in `coaster.py`

### Calibration
- raw_zero_value (float): Raw analog value produced with nothing on the coaster.
- raw_ref_value (float): Raw analog value produced with a reference item on the coaster.
- ref_weight_grams (float): Expected mass (in grams) of the reference item.
- scale_factor (float): Raw analog value of one gram based on zero value and ref. value. 
- last_calibrated (str): Metadata to track when coaster was last calibrated (ISO Format).

### Cup
- name (str): Cup name
- cup_weight_grams (float): Mass (in grams) of the cup when it is empty.

### Profile
- name (str): Profile name.
- daily_goal_ml (float): Amount of water in mL that the user would like to reach. Defaults to 2000.0.
- reminder_frequency (float): Time (in seconds) passed without drinking water to send an alert. Defaults to 30 minutes (1800 seconds).
- recent_cup (Cup): Cup used last time by user. None, if user has no cup saved. 
- last_sip_time (str): Time the user last took a sip (ISO Format).

## File Descriptions

`coaster.py`: Smart Coaster System

`calibration.json`: Stores the calibration settings for the load cell scale.

`coaster_profiles.json`: Stores the profiles for Profile Mode.

`hx711.py`: Driver for the HX711 amplifier used with the load cell.

`calibrate.py`: Most simple program to get a read from the load cell. Used for testing components, not for the Smart Coaster.

# Hardware
Description of the physical system

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
Made by Janet Bai: [OnShape Link](https://cad.onshape.com/documents/2e91cbeae698b6ab89716089/w/e1222bfd9f4ff7c4da097b5e/e/e275416058a4f195d6499203?renderMode=0&uiState=69a8888a4e89299f92843471)
