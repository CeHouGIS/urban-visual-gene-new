"""CPU-only report from recorded predictions and completed training metrics."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from common import write_json
ROOT=Path(__file__).resolve().parents[1]

def summarize():
 c=json.loads((ROOT/'configs/experiment.json').read_text());d=ROOT/'reports/frozen_diagnostics'
 verify=json.loads((d/'checkpoint_verification.json').read_text())
 if not verify['all_state_tensors_equal_checkpoint'] or verify['known_reconstruction_psnr']<25:raise AssertionError('Invalid diagnostic checkpoint')
 interventions=json.loads((d/'intervention_summary.json').read_text());comp=json.loads((d/'composition_summary.json').read_text());recon=json.loads((d/'new_city_reconstruction.json').read_text())
 all_done=all((ROOT/f'runs/{v}/city_confirmation_metrics.json').exists() for v in c['variants'])
 post_done=(ROOT/'reports/trained_composition_controls.json').exists()
 time=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S 北京时间')
 lines=['# 城市视觉基因三项验证结果','',f'更新：{time}。正式诊断已严格加载 step 6000 权重；已知图片回归检查通过。','',
 '## 当前判断','',
 f'- 新城市重建：128 个独立城市，平均 PSNR {recon["psnr"]["mean"]:.2f} dB，SSIM {recon["ssim"]["mean"]:.4f}，MAE {recon["mae"]["mean"]:.5f}。支持重建泛化。',
 '- 干预：不少基因/组产生稳定的低层视觉变化，但相似质量的其他基因组也很稳定；这不构成城市语义或独立生成单元的充分证据。',
 '- 组合：选定基因组的迁移弱，未显示优于其他基因组。全部特征混合能移入中心图案，但会出现明显的拼接边界。当前只能确认编码可编辑，尚不能确认有自然、选择性的“城市视觉基因组合”。',
 ('- 相同编码字节预算的六次训练已完成，分档结果见下表。重建对照也不能替代基因组合验证。' if all_done else '- 相同编码字节预算的三方法比较需等待新的从头训练；旧模型的好重建不能替代这项验证。'),'',
 '## 实验一：同编码预算对照','',
 '三方法 learned / frozen-random / dense，多码率 32,352 / 129,120 / 516,192 字节，两个 seed，共六次训练，每次 6,000 更新、global batch 256。每三步覆盖一轮三档预算，每档实际训练 2,000 更新。网络权重共享后不计入单图码长，稀疏索引和固定容量填充已计入。高档码长超过原始 uint8 RGB，不能宣称图像压缩优势。','',
 '|模型|状态|完成步数|验证平均目标|','|---|---|---:|---:|']
 for variant in c['variants']:
  p=ROOT/f'runs/{variant}/progress.json';r=json.loads(p.read_text()) if p.exists() else {}
  best=r.get('best_validation_loss');lines.append(f'|{variant}|{r.get("status","排队")}|{r.get("step",0)}/6000|{best:.5f}|' if best is not None else f'|{variant}|{r.get("status","排队")}|{r.get("step",0)}/6000|待验证|')
 complete=[v for v in c['variants'] if (ROOT/f'runs/{v}/city_confirmation_metrics.json').exists()]
 if complete:
  lines+=['','新城市分档结果（仅列已完成模型）：','','|模型|字节/图|PSNR|SSIM|LPIPS|','|---|---:|---:|---:|---:|']
  for v in complete:
   ms=json.loads((ROOT/f'runs/{v}/city_confirmation_metrics.json').read_text())
   for m in ms['rates'].values():lines.append(f'|{v}|{m["encoded_bytes_per_image"]}|{m["psnr_mean_per_image"]:.2f}|{m["ssim"]:.4f}|{m["lpips"]:.4f}|')
 lines+=['','## 实验二：跨城市干预','','发现集 64 张固定选择：每尺度两个单基因 + top8 组；确认集 128 个新城市。alpha 为 0、0.5、1、1.5。匹配对照使用不同基因且匹配特征贡献强度，每组各自在发现集估计变化方向。下表为组级结果，完整九目标和逐图数据见 JSON。','','|尺度|组|增强变化 MAE|增强方向余弦|剂量单调比例|','|---|---|---:|---:|---:|']
 for s in ['coarse','medium','fine']:
  for r in interventions:
   if r['target']==s+'_group_top8':lines.append(f'|{s}|{"选定组" if r["kind"]=="selected" else "匹配其他基因组"}|{r["enhance_mae"]["mean"]:.5f}|{r["enhance_direction_cosine"]["mean"]:.3f}|{r["direction_monotonic_fraction"]["mean"]:.1%}|')
 lines+=['','方向余弦衡量亮度、颜色差、对比度、边缘等八个属性，不表示树、建筑等语义。高余弦可能由单一亮度方向主导。删除与增强的方向一致性在部分中尺度组较差；完整记录中保留该失败。其他基因组也能有稳定方向，不能据此排除通用特征分解。','','![跨城市干预](reports/frozen_diagnostics/intervention_summary.png)','','示例固定为名单前四张，无生成质量重抽：','','![细尺度组干预](reports/frozen_diagnostics/fine_group_top8.png)','','## 实验三：A/B 组合','','保留 A 粗尺度，替换中心中/细尺度的 top8 基因表达；32 对不同新城市，配对由原图低层布局决定。局部向 B 靠近幅度定义为 `(编辑前到B的MAE - 编辑后到B的MAE) / 编辑前到B的MAE`，基线使用各自完整重建。中心指标避开边界，区域外指标排除 32px halo。','','|方式|局部向 B 靠近幅度|95% 描述性区间|区域外变化 MAE|','|---|---:|---|---:|']
 labels={'gene_group':'选定基因组','matched_other_gene_group':'匹配的其他基因组','all_feature_channels':'旧字典模型全部特征通道','rgb_blend':'50% 局部 RGB 混合'}
 for key,m in comp['metrics'].items():
  val=m['donor_distance_reduction_fraction'];lo,hi=val['ci95'];lines.append(f'|{labels[key]}|{val["mean"]:.2%}|[{lo:.2%}, {hi:.2%}]|{m["receiver_outside_mae"]["mean"]:.6f}|')
 pairrows=json.loads((d/'composition_per_pair.json').read_text());selected={r['pair']:r['donor_distance_reduction_fraction'] for r in pairrows if r['mode']=='gene_group'};random={r['pair']:r['donor_distance_reduction_fraction'] for r in pairrows if r['mode']=='matched_other_gene_group'}
 from diagnose_genes import ci
 paired=ci([selected[k]-random[k] for k in selected]);write_json(d/'paired_selected_minus_random.json',paired)
 lines+=['',f'选定组减其他基因组的逐对差值均值为 {paired["mean"]:.2%}，bootstrap 95% 区间 [{paired["ci95"][0]:.2%}, {paired["ci95"][1]:.2%}]。该差值仍受组贡献幅度差异影响，不能直接当作因果优劣。','',
 '选定基因组变化小，没有出现清晰的对象级迁移；其他基因组的变化也小。全部通道混合接近在区域内替换图像，供体距离下降更大，但有边界伪影，不能称为自然组合。RGB 混合的 50% 距离改善由其定义决定，不能据此判断视觉真实性。','',
 ('这里的“全部特征通道”来自旧字典模型，并非普通 AE。新六模型的普通 AE 等组合对照已补齐，见 reports/trained_composition_controls.json 与以下曲线。' if post_done else '这里的“全部特征通道”来自旧字典模型，并非普通 AE。新六模型完成后，后台自动补齐相同码率的 learned/random/dense 对照，alpha=0.25/0.5/1，输出供体 LPIPS 改善与区域外扰动曲线。'),'',
 '![组合例子](reports/frozen_diagnostics/composition_example.png)','','[完整32对逐对结果](reports/frozen_diagnostics/composition_per_pair.json) · [全部例图](reports/frozen_diagnostics/composition_overview.png)','','## 输出与复现','','- 方案与假设：[PROTOCOL.md](PROTOCOL.md)。','- 数据选择：`data/discovery_selection.json`、`data/confirmation_selection.json`。','- 诊断：`reports/frozen_diagnostics/`，完整特征码在 `confirmation_codes.pt`。','- 训练：`runs/<method>_s<seed>/`；进度 `queue_status.json`。','- 新训练组合对照：`reports/trained_controls/`；后台状态 `reports/postprocess_status.json`。','- 3 项 CPU 编解码测试、4 项 GPU 模型测试已通过；每组正式训练前必须通过对应双卡 smoke。','- 匿名评审材料已生成，尚无人工评分；没有给出人工真实性结论。','- `reports/INVALID_unloaded_checkpoint_diagnostics` 明确标记为无效，不参与上述结果。','',
 '## 对研究主张的含义','','目前更稳妥的说法是：学到了一套能够重建街景的多尺度加性特征。要进一步称为城市视觉基因，需要在相同信息预算对照和可观测的组合迁移上提供额外优势。稀疏竞争或重建目标本身不保证对象/语义解耦，也不保证混合码位于训练分布内。','']
 if all_done:lines+=['','## 新训练的相同码长对照曲线','','![重建码长对照](reports/rate_distortion_controls.png)']
 if post_done:lines+=['','![组合迁移与区域外扰动](reports/composition_controls_tradeoff.png)','','原始逐对结果在 reports/trained_controls/，按模型/码率/alpha 分组。全码交换用于所有三种模型；字典模型另含选定与匹配的其他基因组交换。全码交换与少量基因交换的传入信息量不同，不能只比供体相似度。']
 (ROOT/'RESULTS.md').write_text('\n'.join(lines))
 # Display only the first two predetermined pairs, with A/B/group/full-channel columns.
 full=Image.open(d/'composition_overview.png').convert('RGB');side=208;height=236
 preview=Image.new('RGB',(side*4,height*2),'white')
 for j,col in enumerate([0,1,2,4]):preview.paste(full.crop((col*side,0,(col+1)*side,height*2)),(j*side,0))
 preview.save(d/'composition_example.png')
 with (d/'blind_ratings_template.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['rater_id','pair','candidate','donor_transfer_0_to_4','receiver_preservation_0_to_4','realism_0_to_4','comments'])
  for p in range(32):
   for candidate in 'ABCD':writer.writerow(['',p,candidate,'','','',''])
 with zipfile.ZipFile(ROOT/'reports/diagnostics_results.zip','w',zipfile.ZIP_DEFLATED) as z:
  for path in [ROOT/'PROTOCOL.md',ROOT/'RESULTS.md',ROOT/'data/confirmation_selection.json',ROOT/'data/discovery_selection.json']+list(d.glob('*.png'))+list(d.glob('*.json'))+[d/'blind_ratings_template.csv']:
   z.write(path,path.relative_to(ROOT))
 print('Updated '+str(ROOT/'RESULTS.md'))
if __name__=='__main__':summarize()
