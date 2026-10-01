# Q4 synthetic-data log

Machine: MacBook, Apple Silicon (arm64), macOS 26 (Darwin 25.6.0), system Python 3.9.6.
Date: 2026-09-30.

## Environment

| step | command | outcome |
| :--- | :--- | :--- |
| create virtual environment | `python3 -m venv q4_data_sim/synthetic/.venv` | success |
| install packages | `.venv/bin/pip install -r requirements.txt` | success: numpy 2.0.2, opencv-python 5.0.0.93, matplotlib 3.9.4, scikit-learn 1.6.1 |
| check OpenCV 5 APIs | `python -c "import cv2; cv2.grabCut(...); cv2.calcOpticalFlowFarneback"` | success: both functions present, `GC_INIT_WITH_RECT = 0`, `GC_EVAL = 2` |

The RA1 and RA2 code runs unchanged under these versions. The scripts import it from `reading_assignments/` through `common.use_reading_assignment`.

## Runs

| script | outcome | run time |
| :--- | :--- | ---: |
| `color_space.py` | success | 0.6 s |
| `gaussian_pyramid.py` | success | 0.5 s |
| `optical_flow.py` | success | 3.4 s |
| `grabcut.py` | success | 10.5 s |
| `supervised_learning.py` | success | 15.7 s |
| `run_all.py` | success, wrote `results/SUMMARY.md` | 30.6 s |

A second `run_all.py` produced identical numbers apart from timings.

## Problems found and how each script handles them

**Color-space threshold tuning was under-determined.** On the unshadowed image every candidate threshold reached IoU 1.000, so the choice among them came down to loop order. With the first-found threshold, CIELAB and RGB both fell to about 0.5 under shadow. With an added rule of keeping the widest accurate box, all three spaces scored 0.98 or higher, because the blue-gray background was far from orange in every color space. The final scene adds red and olive clutter, colors whose RGB values sit between the lit and the shadowed disc at a different hue, and keeps the widest-box rule. Result: HSV 1.000, RGB 0.542, CIELAB 0.534 under shadow.

**Farneback failed on the (12, 9) shift with one documented setting.** With `poly_n=5, poly_sigma=1.1`, endpoint error was 11.77 px on the (12, 9) shift. Changing `levels` (3, 4, 5) or `winsize` (15, 21, 31) did not help. With `poly_n=7, poly_sigma=1.5`, the other pair the OpenCV documentation recommends, the error was 0.004 px. The script uses `poly_n=7` and prints both settings.

**Pseudo-inverse cutoff in the aperture check.** With `rcond=1e-6`, `np.linalg.pinv` inverted the near-zero eigenvalue (ratio 5.2e-6) and returned (2.905, -3.592) for a true motion of (0.6, 0.4). With `rcond=1e-3`, it returned (0.630, 0.362), matching the edge-normal component (0.623, 0.360).

**GrabCut is all or nothing on this data.** Every run settled in iteration 1, either at IoU above 0.99 or at IoU 0 (every pixel labeled background). Variants tried, none of which produced gradual improvement over iterations: a background patch mostly inside the box (three positions, two colors), uncorrelated per-pixel noise, and a looser box (padding 35 px instead of 12 px). The final script reports this behavior directly: one success case, one collapse case, and a sweep with every run plotted.

**Supervised learning started near chance.** With blob strength 0.15 to 0.35 and texture contrast 0.08 to 0.15, all five classifiers scored 0.53 to 0.59 on the held-out set. Raising blob strength to 0.35 to 0.60, lowering texture contrast to 0.05 to 0.10, and using 4000 training patches gave 0.70 to 0.83.

**Spurious matmul warnings.** numpy 2.0 with Apple's Accelerate BLAS prints `RuntimeWarning: divide by zero encountered in matmul` from `sklearn/utils/extmath.py`. The script filters these warnings and asserts that every score is finite.
