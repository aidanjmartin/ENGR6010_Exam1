# ENGR 6010 Exam 1

Aidan J. Martin, ENGR 6010: AI in Robotics, Vanderbilt University, Fall 2026.

This repository holds the four parts of Exam 1. Q1 is a review paper on five visual perception methods for robotics: color spaces, Gaussian pyramids, optical flow, GrabCut, and supervised learning. Q2 through Q4 form one technical note on the Unitree Go2 X quadruped. Q2 sets up the Unitree Python SDK and its examples. Q3 surveys five Go2 demo repositories and runs one of them. Q4 plans data collection from the robot, runs each of the five methods on synthetic data with exact ground truth, and runs the Go2 locomotion simulation from `unitree_rl_gym`. The code from three earlier reading assignments is included unchanged, and Q4 reuses it.

## Layout

```
README.md                    this file
paper/                       Q1 review paper (.docx and .pdf)
reading_assignments/         earlier demo code, unchanged; Q4 imports from it
  ra1_optical_flow/          PCK decomposition, blob centroid, block matching, Lucas-Kanade
  ra2_grabcut/               interactive GrabCut walkthrough and live segmentation
  ra3_model_less_control/    gradient descent, BFGS and a Broyden-updated robot controller
q2_sdk/                      Q2: SDK setup notes, LOG.md, example output
q3_apps/                     Q3: repository survey, chosen demo, LOG.md, screenshots
  REPO_SURVEY.md             the five Q3 repositories and which one to run
q4_data_sim/                 Q4
  data_collection_plan.md    how to collect Go2 X data, and the synthetic stand-in
  synthetic/                 five synthetic-data scripts, shared helpers, LOG.md
  results/                   figures and SUMMARY.md, the collected result tables
  simulation/                unitree_rl_gym notes, LOG.md, plots, GIFs
technical_note/              the Q2-Q4 technical note
```

## How to run

### Q1: review paper

Open [`paper/Exam1_Q1_Review_Paper.pdf`](paper/Exam1_Q1_Review_Paper.pdf). The Word source is beside it.

### Reading assignments

Each demo runs from its own folder. RA2 and RA3 list their packages in `requirements.txt`, and RA3 needs Python 3.10 or later. RA1 needs numpy, opencv-python and matplotlib, which the Q4 environment below provides.

```bash
cd reading_assignments/ra1_optical_flow
python demo3_block_matching_flow.py      # run before demo 4, which imports it

cd reading_assignments/ra2_grabcut
pip install -r requirements.txt && python grabcut.py

cd reading_assignments/ra3_model_less_control
pip install -r requirements.txt && python main.py
```

### Q2: Unitree SDK

To be completed on Linux.

### Q3: Go2 demo applications

[`q3_apps/REPO_SURVEY.md`](q3_apps/REPO_SURVEY.md) compares the five repositories. Running the chosen demo is to be completed on Linux.

### Q4: synthetic data

Requires Python 3.9 or later. Every script runs on a laptop CPU, and all five together take about 30 seconds on an Apple Silicon MacBook.

```bash
cd q4_data_sim/synthetic
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py              # all five scripts; writes ../results/SUMMARY.md
.venv/bin/python optical_flow.py         # or any one script on its own
```

| script | reading | what it measures |
| :--- | :--- | :--- |
| `color_space.py` | [1] | segmentation IoU in RGB, HSV and CIELAB with and without a shadow; hue spread against saturation |
| `gaussian_pyramid.py` | [2] | Laplacian reconstruction error; aliasing with and without the Gaussian blur |
| `optical_flow.py` | [2] | endpoint error of block matching, three Lucas-Kanade variants and Farneback; the aperture problem |
| `grabcut.py` | [5], [6] | IoU per iteration; final IoU against foreground and background color overlap |
| `supervised_learning.py` | [7], [8] | accuracy of five classifier families on held-out and distribution-shifted patches |

Every script uses a fixed random seed, so a rerun reproduces every number. [`q4_data_sim/results/SUMMARY.md`](q4_data_sim/results/SUMMARY.md) collects the tables, and the figures sit beside it. [`q4_data_sim/synthetic/LOG.md`](q4_data_sim/synthetic/LOG.md) records the setup, the runs and the problems found along the way.

### Q4: data collection

[`q4_data_sim/data_collection_plan.md`](q4_data_sim/data_collection_plan.md) describes what the Go2 X exposes over WebRTC, how a Raspberry Pi 4 on the robot can log it with time stamps, and how each synthetic script maps to a real data stream.

### Q4: simulation

To be completed on Linux.

## References

These follow the numbering of the Q1 review paper. The Q4 scripts cite them by number.

1. N. A. Ibraheem, M. M. Hasan, R. Z. Khan, and P. K. Mishra, "Understanding color models: A review," *ARPN Journal of Science and Technology*, vol. 2, no. 3, pp. 265–275, 2012.
2. R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. Springer, 2022.
3. B. Ghanekar, L. R. Johnson, J. L. Laughlin, M. K. O'Malley, and A. Veeraraghavan, "Video-based surgical tool-tip and keypoint tracking using multi-frame context-driven deep learning models," in *Proc. IEEE 22nd Int. Symp. Biomedical Imaging (ISBI)*, 2025.
4. A. J. Martin, J. Atoum, and J. Y. Wu, "Challenges in generalizing vision-based surgical tool keypoint detection across surgical tasks and datasets," submitted to *SPIE Medical Imaging*, 2027.
5. C. Rother, V. Kolmogorov, and A. Blake, "'GrabCut': Interactive foreground extraction using iterated graph cuts," *ACM Transactions on Graphics*, vol. 23, no. 3, pp. 309–314, 2004.
6. B. Peng, L. Zhang, and D. Zhang, "A survey of graph theoretical approaches to image segmentation," *Pattern Recognition*, vol. 46, no. 3, pp. 1020–1038, 2013.
7. S. B. Kotsiantis, "Supervised machine learning: A review of classification techniques," *Informatica*, vol. 31, no. 3, pp. 249–268, 2007.
8. L. Bottou, F. E. Curtis, and J. Nocedal, "Optimization methods for large-scale machine learning," *SIAM Review*, vol. 60, no. 2, pp. 223–311, 2018.
9. University of Southampton, "Interactive optical flow demonstration," Java applet. https://www.southampton.ac.uk/~msn/book/new_demo/opticalFlow/
10. "Network flow visualizer (max flow / min cut)," interactive web tool. https://web-apps.thecoatlessprofessor.com/graph-algorithms/network-flow.html
11. OpenCV, "Interactive foreground extraction using GrabCut algorithm," OpenCV-Python Tutorials.
