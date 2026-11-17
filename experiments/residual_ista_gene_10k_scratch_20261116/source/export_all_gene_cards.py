"""Render every dictionary gene and its five saved spatial expression maps.

CPU visualization only; all data come from checkpoint-verified FP32 exports.
"""
import csv
import html
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use('Agg')
from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


SCALES = ('coarse', 'medium', 'fine')
WIDTH = 1600
COL_WIDTH = 260
LEFT = 20
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTS = {size: ImageFont.truetype(FONT_PATH, size) for size in [12, 14, 16, 18, 20, 24, 28, 32]}


def montage(atom):
    if atom.shape[-1] == 1:
        return atom[:, 0, 0].reshape(8, 8)
    value = np.full((31, 31), np.nan)
    for channel in range(64):
        row, column = divmod(channel, 8)
        value[row * 4:row * 4 + 3, column * 4:column * 4 + 3] = atom[channel]
    return value


def heatmap(values, maximum, cmap_name, side, signed=False):
    normalized = (values / maximum + 1) / 2 if signed else values / maximum
    cmap = colormaps[cmap_name].copy()
    cmap.set_bad('#dddddd')
    rgb = np.rint(cmap(np.ma.masked_invalid(normalized.clip(0, 1)))[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((side, side), Image.Resampling.NEAREST)


def draw_colorbar(canvas, x, y, width, cmap, low, high, label):
    draw = ImageDraw.Draw(canvas)
    ramp = np.linspace(0, 1, width)[None]
    rgb = np.rint(colormaps[cmap](ramp)[:, :, :3] * 255).astype(np.uint8)
    bar = Image.fromarray(rgb).resize((width, 10))
    canvas.paste(bar, (x, y))
    draw.text((x, y + 11), f'{low:.4f}', font=FONTS[14], fill='#374151')
    draw.text((x + width, y + 11), f'{high:.4f}', font=FONTS[14], fill='#374151', anchor='ra')
    draw.text((x + width // 2, y + 11), label, font=FONTS[14], fill='#374151', anchor='ma')


def paste_gene(canvas, y, scale, gene, arrays, limits, samples, side=204, headings=True):
    draw = ImageDraw.Draw(canvas)
    atom = arrays[f'{scale}_dictionary'][gene]
    activations = arrays[f'{scale}_expressions'][:, gene]
    count = int((activations > 0).any((1, 2)).sum())
    top = y + 34
    for column in range(6):
        x = LEFT + column * COL_WIDTH + (COL_WIDTH - side) // 2
        if column == 0:
            label = f'{scale} gene {gene:03d}'
            values = montage(atom)
            tile = heatmap(values, limits[scale]['weights'], 'RdBu_r', side, signed=True)
        else:
            label = f'#{column} {samples[column - 1]["city"]}' if headings else f'Image #{column}'
            values = activations[column - 1]
            tile = heatmap(values, limits[scale]['expression'], 'magma', side)
        draw.text((x, y + 6), label, font=FONTS[18], fill='#111827')
        canvas.paste(tile, (x, top))
        draw.rectangle((x, top, x + side - 1, top + side - 1), outline='#c4cad2')
        if column == 0:
            draw.text((x, top + side + 6), f'D: 64 x {atom.shape[-1]} x {atom.shape[-1]}', font=FONTS[16], fill='#374151')
            draw.text((x, top + side + 28), f'Responds in {count} / 5 images', font=FONTS[16], fill='#374151')
        else:
            locations = int((values > 0).sum())
            draw.text((x, top + side + 6), f'mean {values.mean():.5f}; max {values.max():.5f}', font=FONTS[14], fill='#374151')
            draw.text((x, top + side + 26), f'A > 0: {locations} / {values.size}', font=FONTS[14], fill='#374151')


def page_header(canvas, scale, page_number, genes, arrays, limits, samples):
    draw = ImageDraw.Draw(canvas)
    draw.text((LEFT, 14), f'{scale.upper()} | genes {genes[0]:03d}-{genes[-1]:03d} | page {page_number}', font=FONTS[28], fill='#111827')
    draw.text((LEFT, 56), 'Multiscale ISTA, checkpoint 6000. Every gene shown in ID order, including inactive genes.', font=FONTS[18], fill='#374151')
    draw.text((LEFT, 83), 'Column 1: shared dictionary D. Columns 2-6: actual expression maps A in these five held-out streets.', font=FONTS[18], fill='#374151')
    draw.text((LEFT, 115), 'Red/blue = signed weights. Brightness = expression strength. Fixed linear limits within each scale.', font=FONTS[16], fill='#374151')
    for i, sample in enumerate(samples):
        x = LEFT + (i + 1) * COL_WIDTH + 54
        image = Image.fromarray(np.rint(arrays['originals'][i].transpose(1, 2, 0) * 255).astype(np.uint8))
        canvas.paste(image.resize((152, 152), Image.Resampling.LANCZOS), (x, 176))
        draw.text((x + 76, 145), f'#{i+1} {sample["city"]}', font=FONTS[18], fill='#111827', anchor='ma')
    draw_colorbar(canvas, LEFT + 5, 182, 230, 'RdBu_r', -limits[scale]['weights'], limits[scale]['weights'], 'D')
    draw_colorbar(canvas, LEFT + 5, 243, 230, 'magma', 0, limits[scale]['expression'], 'A')
    draw.text((LEFT + 5, 298), 'A > 0 includes weak values.', font=FONTS[14], fill='#374151')


def write_gallery(destination, entries, limits, metadata):
    cards = []
    for entry in entries:
        scale, gene = entry['scale'], entry['gene_id']
        label = f'{scale}:{gene:03d}'
        safe_path = html.escape(entry['path'], quote=True)
        cards.append(f'<article data-scale="{scale}" data-gene="{gene}" data-count="{entry["responded_image_count"]}">'
                     f'<h2>{label} · 响应 {entry["responded_image_count"]}/5 张</h2>'
                     f'<a href="{safe_path}" target="_blank"><img loading="lazy" src="{safe_path}" '
                     f'width="1600" height="398" alt="{label} 权重及五张街景表达图"></a></article>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISTA · 全部 512 个基因</title><style>
body{margin:0;font:16px system-ui,sans-serif;background:#f3f5f7;color:#17202c}header{background:white;padding:24px max(20px,calc((100vw - 1550px)/2));border-bottom:1px solid #dce1e8}h1{margin:0 0 12px;font-size:28px}p{line-height:1.65;margin:8px 0}.controls{position:sticky;top:0;background:#ffffffee;backdrop-filter:blur(8px);padding:14px 24px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #dce1e8;z-index:2}select,input{font:inherit;padding:7px;border:1px solid #adb6c3;border-radius:6px}main{max-width:1560px;margin:18px auto;padding:0 12px}article{background:white;margin:20px 0;border:1px solid #dce1e8;border-radius:8px;overflow:hidden}h2{font-size:17px;margin:12px 18px}img{width:100%;height:auto;display:block}a{color:#135cad}article[hidden]{display:none}.note{color:#536174;font-size:14px}#empty{padding:40px;text-align:center}</style>
<header><h1>多尺度 ISTA：逐个查看全部 512 个基因</h1><p>同一个第 6000 步检查点。粗尺度 256 个，中尺度 128 个，细尺度 128 个。每张卡左侧是固定的字典权重 D，右侧是同一个 gene 在五张街景中的表达图 A。</p><p>红蓝表示权重的正负；热力图越亮，表达越强。每个尺度内使用相同色标。64 个特征通道按 8×8 拼排仅为展示，不代表图像空间位置。</p><p class="note">图片顺序：Sofia、Tallinn、Tauranga、Cancún、Santo Domingo。响应按 A&gt;0 统计。均值包含零值；表达强度不等于对 RGB 的贡献。</p><p><a href="all_512_gene_cards.pdf">完整 PDF 图册（64 页）</a> · <a href="gene_catalog.csv">完整数值 CSV</a></p></header>
<nav class="controls"><label>尺度 <select id="scale"><option value="all">全部尺度</option><option value="coarse">粗尺度（256）</option><option value="medium">中尺度（128）</option><option value="fine">细尺度（128）</option></select></label><label>gene 编号 <input id="gene" type="number" min="0" max="255" placeholder="全部" style="width:100px"></label><label>响应 <select id="count"><option value="all">全部</option><option value="active">至少一张</option><option value="5">五张全部</option><option value="0">五张均未响应</option></select></label><span id="result"></span></nav><main>'''
    document += '\n'.join(cards)
    document += '''<p id="empty" hidden>没有符合条件的基因。</p></main><script>
const controls=['scale','gene','count'].map(id=>document.getElementById(id));
function filter(){const [scale,gene,count]=controls.map(x=>x.value);let visible=0;for(const card of document.querySelectorAll('article')){const match=(scale==='all'||card.dataset.scale===scale)&&(gene===''||Number(gene)===Number(card.dataset.gene))&&(count==='all'||(count==='active'?Number(card.dataset.count)>0:card.dataset.count===count));card.hidden=!match;if(match)visible++;}document.getElementById('result').textContent=`显示 ${visible} / 512 个`;document.getElementById('empty').hidden=visible>0;}for(const element of controls)element.addEventListener('input',filter);filter();</script></html>'''
    (destination / 'index.html').write_text(document)


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'reports/five_street_expressions_20261117_062418'
    metadata = json.loads((source / 'metadata.json').read_text())
    assert metadata['checkpoint_step'] == 6000
    assert metadata['checkpoint_verification']['all_state_tensors_equal_checkpoint']
    with np.load(source / 'five_street_expression_arrays.npz') as saved:
        arrays = {key:saved[key] for key in saved.files}
    samples = metadata['selected_images']
    assert len(samples) == 5 and arrays['originals'].shape == (5, 3, 256, 256)
    assert sum(len(arrays[f'{s}_dictionary']) for s in SCALES) == 512
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    destination = root / 'reports' / f'all_512_gene_cards_{stamp}'
    destination.mkdir(exist_ok=False)
    limits = {}
    for scale in SCALES:
        dictionary, activation = arrays[f'{scale}_dictionary'], arrays[f'{scale}_expressions']
        assert len(dictionary) == activation.shape[1]
        assert np.isfinite(dictionary).all() and np.isfinite(activation).all()
        assert (activation >= 0).all()
        np.testing.assert_allclose(np.linalg.norm(dictionary.reshape(len(dictionary), -1), axis=1), 1., atol=1e-6)
        limits[scale] = {'weights': float(np.abs(dictionary).max()), 'expression': float(activation.max())}
        (destination / scale).mkdir()
    entries = []
    csv_rows = []
    page_count = 0
    global_id = 0
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.subplots_adjust(0, 0, 1, 1)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis('off')
    with PdfPages(destination / 'all_512_gene_cards.pdf') as pdf:
        pdf.infodict().update(Title='All 512 ISTA genes: weights and five street expression maps', Author='Urban Visual Gene', Subject='Frozen checkpoint 6000; eight genes per page; all genes included.')
        for scale in SCALES:
            dictionary, activation = arrays[f'{scale}_dictionary'], arrays[f'{scale}_expressions']
            for start in range(0, len(dictionary), 8):
                genes = list(range(start, min(start + 8, len(dictionary))))
                page_count += 1
                page = Image.new('RGB', (WIDTH, 2262), 'white')
                page_header(page, scale, page_count, genes, arrays, limits, samples)
                for row_index, gene in enumerate(genes):
                    response_count = int((activation[:, gene] > 0).any((1, 2)).sum())
                    filename = f'{scale}/gene_{gene:03d}.png'
                    card = Image.new('RGB', (WIDTH, 398), 'white')
                    draw = ImageDraw.Draw(card)
                    draw.text((LEFT, 8), f'{scale.upper()} gene {gene:03d}  |  checkpoint 6000  |  responds in {response_count}/5 streets', font=FONTS[24], fill='#111827')
                    paste_gene(card, 44, scale, gene, arrays, limits, samples, side=208)
                    draw_colorbar(card, LEFT + 28, 350, 208, 'RdBu_r', -limits[scale]['weights'], limits[scale]['weights'], 'D')
                    draw_colorbar(card, LEFT + COL_WIDTH + 28, 350, 1180, 'magma', 0, limits[scale]['expression'], 'A: fixed scale-wide range')
                    card.save(destination / filename)
                    entries.append({'global_id': global_id, 'scale': scale, 'gene_id': gene,
                                    'path': filename, 'responded_image_count': response_count})
                    row = {'global_id': global_id, 'scale': scale, 'gene_id': gene,
                           'atom_shape': 'x'.join(map(str, dictionary[gene].shape)), 'responded_image_count': response_count}
                    for index, sample in enumerate(samples):
                        values = activation[index, gene]
                        row[f'image_{index+1}_test_index'] = sample['test_index']
                        row[f'image_{index+1}_positive_locations'] = int((values > 0).sum())
                        row[f'image_{index+1}_mean_expression'] = float(values.mean())
                        row[f'image_{index+1}_max_expression'] = float(values.max())
                    csv_rows.append(row)
                    paste_gene(page, 350 + row_index * 232, scale, gene, arrays, limits, samples, side=148, headings=False)
                    page_draw = ImageDraw.Draw(page)
                    page_draw.line((LEFT, 350 + row_index * 232 + 228, WIDTH - LEFT, 350 + row_index * 232 + 228), fill='#e2e6ec')
                    global_id += 1
                footer = ImageDraw.Draw(page)
                footer.text((LEFT, 2220), 'Mean includes zeros; response counts expression locations. Gene IDs are scale specific. No model execution or training.', font=FONTS[16], fill='#536174')
                axis.clear()
                axis.axis('off')
                axis.imshow(page)
                pdf.savefig(figure, dpi=190)
                if start == 0:
                    page.save(destination / f'{scale}_first_8_genes.png')
                print(json.dumps({'status': 'rendering', 'completed_genes': global_id, 'pdf_pages': page_count}), flush=True)
    plt.close(figure)
    assert len(entries) == len(csv_rows) == 512 and page_count == 64
    assert len({entry['path'] for entry in entries}) == 512
    with (destination / 'gene_catalog.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_gallery(destination, entries, limits, metadata)
    np.savez_compressed(destination / 'actual_gene_weights.npz', **{s:arrays[f'{s}_dictionary'] for s in SCALES})
    with (destination / 'actual_gene_weights.txt').open('w') as stream:
        for scale in SCALES:
            for gene, atom in enumerate(arrays[f'{scale}_dictionary']):
                stream.write(f'\n{scale} gene {gene:03d} shape={atom.shape}\n')
                stream.write(np.array2string(atom, precision=7, threshold=atom.size, max_line_width=130) + '\n')
    (destination / 'README.md').write_text('# 全部 512 个 ISTA 基因\n\n打开 index.html 可按尺度、编号和响应图片数筛选。每个 gene 单独 PNG 保存在 coarse/medium/fine。PDF 按粗、中、细尺度及编号顺序展示全部 gene，每页 8 个，共 64 页。\n\n每张卡最左列为实际归一化字典权重 D：粗尺度 64×1×1，中细尺度 64×3×3，64 个通道按 8×8 拼排。其余五列是此前固定五张街景的实际表达图 A。表达热力图、权重热力图在每个尺度内均采用统一线性色标。红蓝表示权重正负，灰色为卷积核通道间展示间隙；表达图亮度表示系数大小。\n\n图片顺序：Sofia、Tallinn、Tauranga、Cancún、Santo Domingo。mean 包含零，max 为最大表达，A>0 是原生表达网格中的非零位置数。未响应基因亦完整保留。字典图不对应 RGB 外观；表达强度不等于最终重建贡献。\n\nactual_gene_weights.txt 打印全部实际权重数值；actual_gene_weights.npz 为无损数组；gene_catalog.csv 为全部表达统计；metadata.json 记录来源和色标。使用已有第 6000 步检查点严格验证的 FP32 导出结果，CPU 仅绘图和统计，没有新模型推理或训练。\n')
    manifest = {'status': 'complete', 'gene_count': 512, 'individual_png_count': 512, 'pdf_pages': 64,
                'source_arrays': str(source / 'five_street_expression_arrays.npz'),
                'checkpoint': metadata['checkpoint'], 'checkpoint_step': 6000,
                'checkpoint_verification': metadata['checkpoint_verification'],
                'display_limits': limits, 'samples': [{k:s[k] for k in ['city','country','test_index']} for s in samples],
                'entries': entries, 'computation': 'CPU rendering and statistics from existing FP32 export; no model execution.',
                'updated_utc': datetime.now(timezone.utc).isoformat()}
    (destination / 'metadata.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(Path(__file__), destination / 'export_all_gene_cards.py')
    # Verify each generated card can be decoded and that no gene was omitted.
    for entry in entries:
        with Image.open(destination / entry['path']) as card:
            assert card.size == (WIDTH, 398)
            card.verify()
    archive_path = destination / 'all_512_gene_cards.zip'
    files = [path for path in destination.rglob('*') if path.is_file() and path != archive_path]
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(files):
            archive.write(path, arcname=path.relative_to(destination))
    print(json.dumps({'status': 'complete', 'destination': str(destination), 'genes': len(entries), 'pdf_pages': page_count,
                      'archive_size_mb': archive_path.stat().st_size / 1e6}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
