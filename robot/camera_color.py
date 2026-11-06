"""Calibrated spatial channel correction, applied on devbox before rectification."""
import json
from pathlib import Path
import numpy as np


def basis(width, height):
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x = (x - width / 2) / (width / 2)
    y = (y - height / 2) / (height / 2)
    r2 = x*x + y*y
    return np.stack([np.ones_like(x), x, y, r2, r2*r2], axis=-1)


class ColorCorrection:
    def __init__(self, path=None):
        self.path = str(path) if path else None
        self.profile = json.loads(Path(path).read_text()) if path else None
        self.gains = None
        self.size = None

    def __call__(self, frame):
        if self.profile is None:
            return frame
        height, width = frame.shape[:2]
        if self.size != (width, height):
            b = basis(width, height)
            blue = np.clip(np.exp(-b @ np.array(self.profile['log_blue_over_green'], np.float32)), .6, 1.5)
            red = np.clip(np.exp(-b @ np.array(self.profile['log_red_over_green'], np.float32)), .6, 1.5)
            gain = np.stack([blue, np.ones_like(blue), red], axis=-1)
            # Preserve the brightness of a neutral reference rather than changing exposure.
            gain /= (.0722*blue + .7152 + .2126*red)[..., None]
            self.gains, self.size = gain, (width, height)
        return np.clip(frame.astype(np.float32)*self.gains, 0, 255).astype(np.uint8)
