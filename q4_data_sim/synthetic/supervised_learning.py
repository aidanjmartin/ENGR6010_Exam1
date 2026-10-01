"""
Supervised classification of synthetic tool-tip patches, with a distribution
shift test.

Readings: Kotsiantis, "Supervised machine learning: A review of classification
techniques," Informatica, 2007 (paper reference [7]); Bottou, Curtis and
Nocedal, "Optimization methods for large-scale machine learning," SIAM Review,
2018 (reference [8]).

Data. Labeled 32 x 32 grayscale patches. Every patch starts as a smooth
background texture with random brightness and contrast. A positive patch adds
a bright, tool-tip-like Gaussian blob of random size and strength at a random
position. Both classes then receive a slight blur and sensor noise. The label
is exact because the generator decides it.

Method. One classifier from each family in Kotsiantis's review, all from
scikit-learn and all fed the 1024 raw pixel values, standardized with the
training-set mean and spread:

    decision tree            logic-based
    k-nearest neighbors      instance-based
    Gaussian naive Bayes     statistical
    support vector machine   kernel method (RBF kernel)
    multilayer perceptron    neural network, one hidden layer of 64 units,
                             trained by stochastic gradient descent with
                             momentum (the method Bottou et al. analyze)

Report. Accuracy on a held-out test set drawn from the training distribution,
then on a shifted test set: brighter, lower contrast and noisier patches with
the same labeling rule. The drop mirrors the cross-dataset drop in Martin,
Atoum and Wu (reference [4]). The figure also plots the MLP's training loss.

Run:  python supervised_learning.py
Out:  results/supervised_learning_examples.png, results/supervised_learning_accuracy.png
"""

import time
import warnings

import cv2
import numpy as np

import common as C
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# numpy 2 on Apple's Accelerate BLAS raises spurious matmul warnings inside
# scikit-learn; main() asserts that every score is finite instead.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"sklearn\.utils\.extmath")

SIZE = 32
N_TRAIN, N_TEST, N_SHIFT = 4000, 1000, 1000
BLOB_AMPLITUDE = (0.35, 0.60)
TEXTURE_CONTRAST = (0.05, 0.10)

# The shift: brighter, lower contrast, noisier. Labels follow the same rule.
TRAIN_CONDITIONS = {"brightness": 0.0, "contrast": 1.0, "noise": 0.03}
SHIFT_CONDITIONS = {"brightness": 0.12, "contrast": 0.7, "noise": 0.06}


# --- data ------------------------------------------------------------------
def make_patch(rng, positive, brightness, contrast, noise):
    tex = rng.standard_normal((SIZE, SIZE)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), rng.uniform(1.0, 3.0))
    tex /= tex.std() + 1e-8
    img = rng.uniform(0.35, 0.55) + rng.uniform(*TEXTURE_CONTRAST) * tex
    if positive:
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        cx, cy = rng.uniform(6, SIZE - 6, size=2)
        s = rng.uniform(1.5, 3.0)
        img = img + rng.uniform(*BLOB_AMPLITUDE) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8)
    img = 0.45 + contrast * (img - 0.45) + brightness
    img = img + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def make_dataset(rng, n, conditions):
    y = rng.permutation(np.arange(n) % 2)
    X = np.stack([make_patch(rng, bool(label), **conditions) for label in y])
    return X, y


# --- models ----------------------------------------------------------------
def make_models():
    return {
        "decision tree": DecisionTreeClassifier(random_state=C.SEED),
        "k-nearest neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian naive Bayes": GaussianNB(),
        "SVM (RBF kernel)": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "MLP (64 hidden, SGD)": MLPClassifier(
            hidden_layer_sizes=(64,), solver="sgd", learning_rate_init=0.01,
            momentum=0.9, batch_size=64, max_iter=150, random_state=C.SEED),
    }


# --- figures ---------------------------------------------------------------
def plot_examples(X_test, y_test, X_shift, y_shift, name):
    fig, axes = plt.subplots(2, 8, figsize=(13, 4.4))
    for row, (X, y, label) in enumerate(((X_test, y_test, "Training distribution"),
                                         (X_shift, y_shift, "Shifted test set"))):
        pos = np.flatnonzero(y == 1)[:4]
        neg = np.flatnonzero(y == 0)[:4]
        for k, idx in enumerate(np.concatenate([pos, neg])):
            ax = axes[row, k]
            ax.imshow(X[idx], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            C.image_axes(ax, "tip" if y[idx] else "no tip")
        axes[row, 0].set_ylabel(label, fontsize=11, color=C.INK, labelpad=8)
    fig.suptitle("Synthetic 32 x 32 patches: the shifted set is brighter, "
                 "lower in contrast and noisier")
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.6)
    C.save_figure(fig, name)


def plot_accuracy(results, loss_curve, name):
    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    names = list(results)
    ys = np.arange(len(names))[::-1]
    for y, n in zip(ys, names):
        a, b = results[n]["test"], results[n]["shift"]
        ax.plot([b, a], [y, y], color=C.AXIS, lw=3, zorder=1)
        ax.scatter([a], [y], s=90, color=C.BLUE, zorder=3, edgecolor="white", lw=1.5)
        ax.scatter([b], [y], s=90, color=C.ORANGE, zorder=3, edgecolor="white", lw=1.5)
        ax.annotate(f"{a:.3f}", (a, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, color=C.INK)
        ax.annotate(f"{b:.3f}", (b, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, color=C.INK)
    ax.scatter([], [], s=90, color=C.BLUE, label="held-out test, same distribution")
    ax.scatter([], [], s=90, color=C.ORANGE, label="shifted test set")
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_xlabel("accuracy")
    lo = min(min(r["shift"] for r in results.values()), 0.5)
    ax.set_xlim(lo - 0.04, 1.02)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.axvline(0.5, color=C.MUTED, lw=1, ls=":")
    ax.text(0.505, -0.55, "chance", fontsize=9.5, color=C.MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.set_title("Test accuracy before and after the shift")
    C.style_axes(ax, grid_axis="x")

    ax = fig.add_subplot(gs[0, 1])
    epochs = np.arange(1, len(loss_curve) + 1)
    ax.plot(epochs, loss_curve, color=C.BLUE, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (cross-entropy)")
    ax.set_ylim(0, max(loss_curve) * 1.05)
    ax.annotate(f"{loss_curve[-1]:.4f} after {len(loss_curve)} epochs",
                (epochs[-1], loss_curve[-1]), textcoords="offset points",
                xytext=(-4, 14), ha="right", fontsize=10, color=C.INK)
    ax.set_title("MLP training loss (SGD with momentum)")
    C.style_axes(ax, grid_axis="both")

    drops = {n: r["test"] - r["shift"] for n, r in results.items()}
    worst = max(drops, key=drops.get)
    fig.suptitle(f"All five classifiers lose accuracy under the shift; "
                 f"{worst} drops most, {results[worst]['test']:.2f} to "
                 f"{results[worst]['shift']:.2f}"
                 if all(d > 0 for d in drops.values()) else
                 f"{worst} drops most under the shift, "
                 f"{results[worst]['test']:.2f} to {results[worst]['shift']:.2f}")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.2)
    C.save_figure(fig, name)


# --- main ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(C.SEED)
    X_train, y_train = make_dataset(rng, N_TRAIN, TRAIN_CONDITIONS)
    X_test, y_test = make_dataset(rng, N_TEST, TRAIN_CONDITIONS)
    X_shift, y_shift = make_dataset(rng, N_SHIFT, SHIFT_CONDITIONS)
    flat = lambda X: X.reshape(len(X), -1)

    C.print_heading("Data")
    C.print_table(["set", "patches", "positives", "brightness offset",
                   "contrast", "noise sigma"],
                  [["train", N_TRAIN, int(y_train.sum()), "0", "1.0",
                    TRAIN_CONDITIONS["noise"]],
                   ["held-out test", N_TEST, int(y_test.sum()), "0", "1.0",
                    TRAIN_CONDITIONS["noise"]],
                   ["shifted test", N_SHIFT, int(y_shift.sum()),
                    f"+{SHIFT_CONDITIONS['brightness']}",
                    SHIFT_CONDITIONS["contrast"], SHIFT_CONDITIONS["noise"]]])
    plot_examples(X_test, y_test, X_shift, y_shift,
                  "supervised_learning_examples.png")

    results, loss_curve = {}, None
    for name, model in make_models().items():
        pipe = make_pipeline(StandardScaler(), model)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(flat(X_train), y_train)
        fit_s = time.perf_counter() - t0
        results[name] = {
            "train": pipe.score(flat(X_train), y_train),
            "test": pipe.score(flat(X_test), y_test),
            "shift": pipe.score(flat(X_shift), y_shift),
            "fit_s": fit_s,
        }
        assert all(np.isfinite(v) for v in results[name].values())
        if isinstance(model, MLPClassifier):
            loss_curve = model.loss_curve_

    C.print_heading("Accuracy by classifier")
    C.print_table(["classifier", "train", "held-out test", "shifted test",
                   "drop", "fit time (s)"],
                  [[n, f"{r['train']:.3f}", f"{r['test']:.3f}",
                    f"{r['shift']:.3f}", f"{r['test'] - r['shift']:+.3f}",
                    f"{r['fit_s']:.2f}"] for n, r in results.items()])
    print(f"MLP training loss: {loss_curve[0]:.4f} after epoch 1, "
          f"{loss_curve[-1]:.4f} after epoch {len(loss_curve)}.")
    plot_accuracy(results, loss_curve, "supervised_learning_accuracy.png")


if __name__ == "__main__":
    main()
