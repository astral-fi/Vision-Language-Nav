"""Contact sheet of an episode: 12 evenly spaced step frames, 4 per row."""
import glob, sys
import cv2, numpy as np
src, out = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"{src}/step_*.jpg"))
pick = [fs[i] for i in np.linspace(0, len(fs) - 1, min(12, len(fs))).astype(int)]
tiles = [cv2.putText(cv2.resize(cv2.imread(f), (320, 240)), f[-8:-4], (8, 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2) for f in pick]
tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
print(f"{len(fs)} frames -> {out}")
