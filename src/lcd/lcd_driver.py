"""
Hardware abstraction for LCD display and touch input.

Wraps Adafruit ILI9341 SPI display and FocalTouch I2C touchscreen.
Gracefully handles missing hardware (returns unavailable).
"""

import logging
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

WIDTH = 320
HEIGHT = 240


class LCDDriver:
    """
    Hardware driver for ILI9341 LCD with FocalTouch input.

    Usage:
        if LCDDriver.available():
            driver = LCDDriver()
            driver.push(image)
            touch = driver.get_touch()
    """

    @classmethod
    def available(cls) -> bool:
        """Check if LCD hardware libraries are available."""
        try:
            import board
            import adafruit_rgb_display.ili9341
            import adafruit_focaltouch
            return True
        except (ImportError, NotImplementedError, RuntimeError):
            return False

    def __init__(self):
        """Initialize SPI display and I2C touch controller."""
        import digitalio
        import board
        import busio
        import adafruit_rgb_display.ili9341 as ili9341
        import adafruit_focaltouch

        cs_pin = digitalio.DigitalInOut(board.CE0)
        dc_pin = digitalio.DigitalInOut(board.D25)
        reset_pin = digitalio.DigitalInOut(board.D24)

        spi = board.SPI()
        i2c = busio.I2C(board.SCL_1, board.SDA_1)

        self._ft = adafruit_focaltouch.Adafruit_FocalTouch(i2c, debug=False)
        self._disp = ili9341.ILI9341(
            spi, rotation=90,
            cs=cs_pin, dc=dc_pin, rst=reset_pin, baudrate=24000000,
        )

        logger.info(f"LCD initialized: {WIDTH}x{HEIGHT}")

    def push(self, image: Image.Image) -> None:
        """Send a PIL Image to the display."""
        self._disp.image(image)

    def get_touch(self) -> Optional[Tuple[int, int]]:
        """
        Poll touch input.

        Returns:
            (x, y) in display coordinates, or None if no touch.
        """
        if self._ft.touched:
            touches = self._ft.touches
            if touches and len(touches) > 0:
                t = touches[0]
                # FocalTouch reports in sensor coords; display rotated 90 deg
                raw_x = t["x"]
                raw_y = t["y"]
                disp_x = raw_y
                disp_y = HEIGHT - raw_x
                return disp_x, disp_y
        return None

    def shutdown(self) -> None:
        """Clean up display resources."""
        try:
            # Clear display to black
            black = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
            self.push(black)
        except Exception:
            pass
