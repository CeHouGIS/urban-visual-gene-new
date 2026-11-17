"""Draw the hidden-state 3D geometry experiment schematic."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/images/v02_coordinate_intervention.png"
FONT = Path("/root/.cache/streetspatial_fonts/NotoSansCJKsc-Regular.otf")


def setup(ax, title):
    ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=12, color="#1d2733")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def camera(ax, x, y, angle, color, label):
    ax.scatter([x], [y], s=100, color=color, edgecolor="white", linewidth=1.3, zorder=5)
    ax.annotate("", xy=(x + 0.12, y + angle), xytext=(x, y),
                arrowprops={"arrowstyle": "-|>", "lw": 2, "color": color}, zorder=4)
    ax.text(x, y - 0.07, label, ha="center", va="top", fontsize=8.5, color=color)


def repeat_capture(ax):
    setup(ax, "1  同地点跨时间：几何稳定性")
    ax.plot([0.08, 0.93], [0.33, 0.33], color="#697887", lw=4, alpha=0.22, solid_capstyle="round")
    point = (0.78, 0.71)
    ax.scatter([point[0]], [point[1]], marker="s", s=155, color="#3d9274", edgecolor="white", zorder=5)
    ax.text(point[0], point[1] + 0.10, "固定建筑角点", ha="center", fontsize=9, color="#252c36")
    p0, p1 = (0.25, 0.33), (0.34, 0.35)
    for p, c, lab, a in [(p0, "#3287b9", "日期 T1", 0.15), (p1, "#cf7c24", "日期 T2", 0.23)]:
        ax.plot([p[0], point[0]], [p[1], point[1]], color=c, lw=1.6, alpha=0.7)
        camera(ax, *p, a, c, lab)
    ax.text(0.07, 0.07, "光照 / 天气可变\n重建后的固定点 XYZ 应稳定", fontsize=9.5,
            color="#39495a", linespacing=1.5)


def multi_view(ax):
    setup(ax, "2  沿街多视点：恢复真实深度")
    ax.plot([0.07, 0.94], [0.43, 0.43], color="#697887", lw=5, alpha=0.22, solid_capstyle="round")
    poses = [(0.14, "#3287b9", "P0"), (0.36, "#cf7c24", "P1"), (0.58, "#7868a8", "P2")]
    near, far = (0.65, 0.72), (0.90, 0.84)
    for x, color, _ in poses:
        ax.plot([x, near[0]], [0.43, near[1]], color=color, lw=1.5, alpha=0.75)
        ax.plot([x, far[0]], [0.43, far[1]], color=color, lw=1.2, alpha=0.5, linestyle="--")
    for x, color, label in poses:
        camera(ax, x, 0.43, 0.01, color, label)
    ax.scatter([near[0]], [near[1]], marker="^", s=150, color="#3d9274", edgecolor="white", zorder=5)
    ax.text(near[0], near[1] + 0.09, "近处路灯", ha="center", fontsize=9, color="#252c36")
    ax.scatter([far[0]], [far[1]], marker="s", s=150, color="#cf3f4f", edgecolor="white", zorder=5)
    ax.text(far[0], far[1] + 0.09, "远处墙角", ha="center", fontsize=9, color="#252c36")
    ax.text(0.07, 0.07, "多视角射线交会 → 局部 3D 点云\n没有平移基线就没有可靠 Z 真值", fontsize=9.5,
            color="#39495a", linespacing=1.5)


def latent_axes(ax):
    setup(ax, "3  隐藏状态：找出并检验 XYZ 子空间")
    ax.add_patch(patches.FancyBboxPatch((0.08, 0.80), 0.82, 0.10,
                    boxstyle="round,pad=0.01,rounding_size=0.012", facecolor="#edf2f5",
                    edgecolor="#c8d3dd", linewidth=1.1))
    ax.text(0.49, 0.85, "视觉 Token 激活 h", ha="center", va="center", fontsize=10,
            color="#28394a", weight="bold")
    ax.annotate("", xy=(0.49, 0.69), xytext=(0.49, 0.78),
                arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": "#697887"})
    ax.text(0.49, 0.72, "投影到候选 rank-3 子空间 U", ha="center", va="bottom", fontsize=8.7,
            color="#8c4d13")

    origin = (0.28, 0.24)
    bx, by, bz = (0.47, 0.02), (0.17, 0.30), (-0.08, 0.34)
    for vec, label, color in [(bx, "X", "#3287b9"), (by, "Y", "#3d9274"), (bz, "Z", "#cf7c24")]:
        ax.annotate("", xy=(origin[0] + vec[0], origin[1] + vec[1]), xytext=origin,
                    arrowprops={"arrowstyle": "-|>", "lw": 2, "color": color})
        ax.text(origin[0] + vec[0] * 1.08, origin[1] + vec[1] * 1.08, label,
                fontsize=10, color=color, weight="bold")

    def project(x, y, z):
        return (origin[0] + x * bx[0] + y * by[0] + z * bz[0],
                origin[1] + x * bx[1] + y * by[1] + z * bz[1])

    a, b = project(0.22, 0.26, 0.70), project(0.78, 0.62, 0.18)
    ax.scatter([a[0]], [a[1]], s=85, color="#3287b9", edgecolor="white", zorder=5)
    ax.scatter([b[0]], [b[1]], s=85, color="#cf3f4f", edgecolor="white", zorder=5)
    ax.text(a[0] - 0.03, a[1] + 0.035, "源点 A", fontsize=8, color="#3287b9")
    ax.text(b[0] + 0.01, b[1] - 0.04, "供体 B", fontsize=8, color="#cf3f4f")
    ax.annotate("", xy=b, xytext=a,
                arrowprops={"arrowstyle": "-|>", "lw": 1.8, "linestyle": "--", "color": "#8c4d13"})
    ax.text(0.66, 0.10, "后层独立 XYZ 解码\n检查内部状态是否移向 B", ha="center",
            fontsize=8.5, color="#39495a")


def main():
    font_manager.fontManager.addfont(str(FONT))
    font_name = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(15, 7.2), facecolor="#fbfcfd")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.05],
                            left=0.045, right=0.97, top=0.78, bottom=0.12, wspace=0.16)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    repeat_capture(axes[0])
    multi_view(axes[1])
    latent_axes(axes[2])

    fig.suptitle("直接观察 VLM 内部几何：视觉 Token → 三维子空间 → 后层激活", x=0.045, y=0.95,
                 ha="left", fontsize=18, weight="bold", color="#17212c")
    fig.text(0.045, 0.865,
             "研究对象是隐藏状态与三维点云对齐；不以模型生成的空间回答作为数据。",
             ha="left", fontsize=11, color="#526171")
    fig.add_artist(patches.FancyBboxPatch((0.045, 0.045), 0.92, 0.045,
                    boxstyle="round,pad=0.008,rounding_size=0.01", transform=fig.transFigure,
                    facecolor="#f0f4f7", edgecolor="#d7dfe7", linewidth=1))
    fig.text(0.505, 0.067,
             "核心判据：rank-3 跨地点泛化 + 跨视角 XYZ 一致 + 子空间干预改变后层内部几何状态。",
             ha="center", va="center", fontsize=10, color="#425263")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
