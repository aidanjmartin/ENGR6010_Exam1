# Q4 synthetic-data results

`synthetic/run_all.py` generates this file. Every number below comes from synthetic data with exact ground truth and a fixed random seed (6010), so a rerun reproduces it. Figures are in this folder.

Environment: Python 3.9.6 on Darwin arm64. Total run time: 29.6 s.

References follow the numbering of the Q1 review paper.

## Color spaces

Script: `synthetic/color_space.py`. Reading: Ibraheem et al. [1]. Run time: 0.6 s.

### Color-space thresholds, tuned on the unshadowed image

| color space | channels | bounds | percentile | margin |
| :--- | :--- | :--- | ---: | ---: |
| RGB | R, G, B | [0.46, 1.05], [0.22, 0.57], [0.01, 0.19] | 0.5 | 1 |
| HSV | H, S | [12.14, 41.54], [0.69, 1.04] | 0.5 | 2 |
| CIELAB | a*, b* | [17.77, 46.59], [37.77, 70.58] | 2 | 1 |


### IoU against the true disc mask

| color space | unshadowed IoU | shadowed IoU | change |
| :--- | ---: | ---: | ---: |
| RGB | 1.000 | 0.542 | -0.458 |
| HSV | 1.000 | 1.000 | +0.000 |
| CIELAB | 1.000 | 0.534 | -0.466 |

HSV hue and saturation do not change when all three channels scale together, so the shadow leaves them in place. CIELAB a* and b* shrink toward zero as L* falls, so the shadowed disc leaves the a*, b* box.

Figure saved: `results/color_space_segmentation.png`

### Hue spread on a noisy gray-to-orange ramp

| saturation | hue std (deg) |
| :--- | ---: |
| 0.00 | 166.0 |
| 0.10 | 13.5 |
| 0.20 | 6.3 |
| 0.30 | 4.1 |
| 0.40 | 3.1 |
| 0.50 | 2.5 |


Figure saved: `results/color_space_hue_instability.png`

## Gaussian pyramids

Script: `synthetic/gaussian_pyramid.py`. Reading: Szeliski Sec. 9.1.1 [2]. Run time: 0.5 s.

### Laplacian pyramid reconstruction (4 levels, float64)

| quantity | value |
| :--- | ---: |
| image size | 256 x 256 |
| max absolute error | 0.00e+00 |
| RMS error | 0.00e+00 |
| image range | [0, 1] |


Figure saved: `results/gaussian_pyramid_levels.png`

### Aliasing: downsampled image vs ideal low-pass result

Fine stripes at 0.406 cycles/px; the limit after 2x downsampling is 0.25 cycles/px.

| factor | RMSE, no blur | RMSE, Gaussian pyramid | ratio |
| :--- | ---: | ---: | ---: |
| 2x | 0.1414 | 0.0023 | 62.0x |
| 4x | 0.1414 | 0.0063 | 22.5x |
| 8x | 0.1415 | 0.0203 | 7.0x |


Figure saved: `results/gaussian_pyramid_aliasing.png`

### Coarse-to-fine: a 15 px motion at each level

| level | image size | motion (px) |
| :--- | ---: | ---: |
| 0 | 256 x 256 | 15 |
| 1 | 128 x 128 | 7.5 |
| 2 | 64 x 64 | 3.75 |
| 3 | 32 x 32 | 1.875 |

Lucas-Kanade linearizes the image around the current estimate, which holds for about 1-2 px of motion. Only the coarsest level brings 15 px inside that range.

## Optical flow

Script: `synthetic/optical_flow.py`. Reading: Szeliski Ch. 9 [2]. Run time: 3.3 s.

### Endpoint error (px) on three known translations

Block matching searches +/-13 px; Farneback is scored on every pixel more than 20 px from the border.

| method | (3, 2) | (2.5, 1.5) | (12, 9) |
| :--- | ---: | ---: | ---: |
| block matching | 0.000 | 0.707 | 0.000 |
| LK single-pass | 1.913 | 1.161 | 14.972 |
| LK iterative | 0.000 | 0.024 | 15.113 |
| LK pyramidal | 0.000 | 0.023 | 0.000 |
| Farneback (dense) | 0.001 | 0.018 | 0.004 |

Lowest fraction of points with an answer, over all methods and shifts: 100%.

### Farneback sensitivity to its polynomial neighborhood

| setting | (3, 2) | (2.5, 1.5) | (12, 9) |
| :--- | ---: | ---: | ---: |
| poly_n=5, poly_sigma=1.1 | 0.000 | 0.017 | 11.769 |
| poly_n=7, poly_sigma=1.5 | 0.001 | 0.018 | 0.004 |

Both settings appear in the OpenCV documentation. The 5-pixel polynomial fit fails on the large shift, so the other results use poly_n=7.

Figure saved: `results/optical_flow_epe.png`

### Dense flow: rotation by 4 degrees and scaling by 1.03

| quantity | value |
| :--- | ---: |
| largest true motion | 15.39 px |
| Farneback mean EPE (interior) | 0.150 px |
| Farneback 95th-percentile EPE | 0.295 px |


Figure saved: `results/optical_flow_dense.png`

### Aperture check: structure tensor eigenvalues (15 x 15 window)

| patch | larger eigenvalue | smaller eigenvalue | ratio | RA1 guard (min 0.0001) |
| :--- | ---: | ---: | ---: | ---: |
| textured patch | 0.916 | 0.587 | 6.4e-01 | solve |
| straight edge | 1.18 | 6.1e-06 | 5.2e-06 | skip |
| flat patch | 0 | 0 | undefined | skip |

| flow vector | u | v |
| :--- | ---: | ---: |
| true motion | 0.600 | 0.400 |
| component normal to the edge | 0.623 | 0.360 |
| Lucas-Kanade, pseudo-inverse | 0.630 | 0.362 |

On the edge, Lucas-Kanade can only return the normal component: motion along the edge leaves the image unchanged.

## GrabCut

Script: `synthetic/grabcut.py`. Reading: Rother et al. [5]; Peng et al. [6]. Run time: 9.6 s.
Bounding box (50, 26, 147, 125) (x, y, w, h); the box alone scores IoU 0.456. Energy is RA2's re-evaluation of the GrabCut energy from the current mask, since OpenCV does not report it. At iteration 1, pixels changed counts the pixels labeled foreground.

### IoU per iteration, success case (d = 2.5, overlap 0.21)

| iteration | IoU | pixels changed | energy |
| :--- | ---: | ---: | ---: |
| 1 | 0.9999 | 8376 | -226895 |
| 2 | 0.9999 | 0 | -227286 |
| 3 | 0.9999 | 0 | -227729 |
| 4 | 0.9999 | 0 | -228094 |
| 5 | 0.9999 | 0 | -228352 |
| 6 | 0.9999 | 0 | -228573 |
| 7 | 0.9999 | 0 | -228736 |
| 8 | 0.9999 | 0 | -228888 |
| 9 | 0.9999 | 0 | -229015 |
| 10 | 0.9999 | 0 | -229151 |


### IoU per iteration, collapse case (d = 1.5, overlap 0.45)

| iteration | IoU | pixels changed | energy |
| :--- | ---: | ---: | ---: |
| 1 | 0.0000 | 0 | -227979 |
| 2 | 0.0000 | 0 | -228129 |
| 3 | 0.0000 | 0 | -228288 |
| 4 | 0.0000 | 0 | -228482 |
| 5 | 0.0000 | 0 | -228693 |
| 6 | 0.0000 | 0 | -229034 |
| 7 | 0.0000 | 0 | -229418 |
| 8 | 0.0000 | 0 | -229940 |
| 9 | 0.0000 | 0 | -230460 |
| 10 | 0.0000 | 0 | -230968 |


Figure saved: `results/grabcut_iterations.png`

### Final IoU after 8 iterations vs color overlap

| separation d | overlap | mean IoU | runs with IoU >= 0.9 | runs with IoU <= 0.1 |
| :--- | ---: | ---: | ---: | ---: |
| 0 | 1.000 | 0.000 | 0 of 5 | 5 of 5 |
| 0.5 | 0.803 | 0.000 | 0 of 5 | 5 of 5 |
| 1 | 0.617 | 0.000 | 0 of 5 | 5 of 5 |
| 1.25 | 0.532 | 0.000 | 0 of 5 | 5 of 5 |
| 1.5 | 0.453 | 0.000 | 0 of 5 | 5 of 5 |
| 1.75 | 0.382 | 0.000 | 0 of 5 | 5 of 5 |
| 2 | 0.317 | 0.998 | 5 of 5 | 0 of 5 |
| 2.25 | 0.261 | 0.999 | 5 of 5 | 0 of 5 |
| 2.5 | 0.211 | 1.000 | 5 of 5 | 0 of 5 |
| 3 | 0.134 | 1.000 | 5 of 5 | 0 of 5 |
| 4 | 0.046 | 1.000 | 5 of 5 | 0 of 5 |
| 5 | 0.012 | 1.000 | 5 of 5 | 0 of 5 |

No run ends between IoU 0.1 and 0.9: each one either separates the blob or collapses to an empty mask.

Figure saved: `results/grabcut_overlap_sweep.png`

## Supervised learning

Script: `synthetic/supervised_learning.py`. Reading: Kotsiantis [7]; Bottou et al. [8]. Run time: 15.7 s.

### Data

| set | patches | positives | brightness offset | contrast | noise sigma |
| :--- | ---: | ---: | ---: | ---: | ---: |
| train | 4000 | 2000 | 0 | 1.0 | 0.03 |
| held-out test | 1000 | 500 | 0 | 1.0 | 0.03 |
| shifted test | 1000 | 500 | +0.12 | 0.7 | 0.06 |


Figure saved: `results/supervised_learning_examples.png`

### Accuracy by classifier

| classifier | train | held-out test | shifted test | drop | fit time (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| decision tree | 1.000 | 0.828 | 0.575 | +0.253 | 8.30 |
| k-nearest neighbors | 0.777 | 0.726 | 0.620 | +0.106 | 0.02 |
| Gaussian naive Bayes | 0.679 | 0.696 | 0.526 | +0.170 | 0.03 |
| SVM (RBF kernel) | 0.925 | 0.818 | 0.682 | +0.136 | 1.49 |
| MLP (64 hidden, SGD) | 1.000 | 0.784 | 0.681 | +0.103 | 0.57 |

MLP training loss: 0.6079 after epoch 1, 0.0023 after epoch 57.

Figure saved: `results/supervised_learning_accuracy.png`
