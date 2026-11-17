"""Show one recorded validation example with plain Chinese labels."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "outputs/v2"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image

font = FontProperties(fname=str(OUTPUT / "report_assets/chinese_explainer_subset.ttf"))
records = [json.loads(line) for line in (OUTPUT / "data/synthetic_20261109/images.val_seen.jsonl").read_text().splitlines()]
sample = np.load(OUTPUT / "m1_dinov3_cpu_spatial_seed_20261109/validation/sample_0001.npz")

def colored(labels):
    palette = np.array([[.36, .57, .91], [.66, .43, .85], [.96, .65, .24], [.28, .75, .58], [.92, .42, .52], [.87, .81, .26]])
    rgb = np.zeros((*labels.shape, 3)) + .10
    for i, value in enumerate(v for v in np.unique(labels) if v):
        rgb[labels == value] = palette[i % len(palette)]
    return rgb

fig, axes = plt.subplots(1, 3, figsize=(12, 5.6))
fig.patch.set_facecolor("#faf9f6")
images = [np.asarray(Image.open(records[1]["image"]).convert("RGB")), colored(sample["gt"]), colored(sample["prediction"])]
titles = ["输入图片", "希望得到的分组", "模型实际输出"]
captions = ["简化合成街景", "每栋房子每棵树都应独立", "还没学会分开"]
for ax, image, title, caption in zip(axes, images, titles, captions):
    ax.imshow(image, interpolation="nearest")
    ax.set_title(title, fontproperties=font, fontsize=19, pad=14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(caption, fontproperties=font, fontsize=15, labelpad=13)
fig.suptitle("现在卡在自动发现景观单元这一步", fontproperties=font, fontsize=23, y=.96)
fig.text(.5, .12, "颜色表示一个候选单元", fontproperties=font, ha="center", fontsize=14)
fig.text(.5, .055, "参考分组仅用于检查  不参与训练", fontproperties=font, ha="center", fontsize=12, color="#555555")
fig.subplots_adjust(left=.025, right=.975, bottom=.23, top=.83, wspace=.11)
target = OUTPUT / "m1_result_plain_chinese_20261109.png"
fig.savefig(target, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print(target)
