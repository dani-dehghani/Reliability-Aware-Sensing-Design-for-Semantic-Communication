from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]


def box(ax, cx, cy, w, h, label, sub=""):
    rect = patches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor="black", facecolor="#eef3fb",
    )
    ax.add_patch(rect)
    ax.text(cx, cy + (0.06 if sub else 0), label, ha="center", va="center",
            fontsize=13, fontweight="bold")
    if sub:
        ax.text(cx, cy - 0.16, sub, ha="center", va="center", fontsize=8.5, color="#444")


def arrow(ax, x0, x1, y, label, color, label_y_offset=0.22):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color))
    ax.text((x0 + x1) / 2, y + label_y_offset, label, ha="center", va="bottom",
            fontsize=9, color=color)


def note(ax, x, y_from, y_text, text, color, va):
    ax.annotate(text, xy=(x, y_from), xytext=(x, y_text), ha="center", va=va, fontsize=8.5, color=color,
                arrowprops=dict(arrowstyle="-", lw=1, color=color, linestyle="dashed"))


def main():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    y = 2.0
    positions = [1.2, 3.8, 6.4, 9.0]
    labels = ["X", "Y_u", "Z", "M"]
    subs = ["true state\n(range, velocity)", "noisy sensor reading",
            "semantic token\n(2 bits)", "received symbol"]

    for cx, lab, sub in zip(positions, labels, subs):
        box(ax, cx, y, 1.3, 0.9, lab, sub)

    arrow(ax, positions[0] + 0.65, positions[1] - 0.65, y, "sensing model", "tab:blue")
    arrow(ax, positions[1] + 0.65, positions[2] - 0.65, y, "sign extractor", "black")
    arrow(ax, positions[2] + 0.65, positions[3] - 0.65, y, "channel W", "tab:red")

    x_sense = (positions[0] + positions[1]) / 2
    x_chan = (positions[2] + positions[3]) / 2
    note(ax, x_sense, y, y - 1.2, "(a) sensing design: alpha splits\nprecision between range and velocity", "tab:blue", "top")
    note(ax, x_chan, y, y - 1.2, "(b) channel mismatch: receiver\nuses W_hat, actual channel is W",
         "tab:red", "top")
    note(ax, x_chan, y + 0.5, y + 1.1, "(c) probing: known tokens Z sent\nthrough W give estimate W_tilde",
         "tab:green", "bottom")

    ax.text(4.0, 3.6, "X -> Y_u -> Z -> M", ha="center", fontsize=11, style="italic", color="#555")

    fig.tight_layout()
    out = ROOT / "figures" / "fig1_system_diagram.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved figures/fig1_system_diagram.png")


if __name__ == "__main__":
    main()
