"""Download only required fp16 SDXL components with pinned SHA256 checks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    metadata = json.loads((ROOT / 'sdxl_repo_files.json').read_text())
    model = metadata['model']
    files = metadata['listing']['Data']['Files']
    revision = next(f['Revision'] for f in files if f['Path'] == 'model_index.json')
    selected = []
    for f in files:
        name = f['Path']
        if f['Type'] != 'blob':
            continue
        if name in ['model_index.json', 'README.md', 'LICENSE.md'] or (
            name.split('/')[0] in ['unet', 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'scheduler']
            and (name.endswith('.json') or name.endswith('merges.txt') or name.endswith('.fp16.safetensors'))
        ):
            selected.append(f)
    out = ROOT / 'models/sdxl-base-1.0'
    out.mkdir(parents=True, exist_ok=True)
    total = sum(f['Size'] for f in selected)
    print(json.dumps({'model': model, 'revision': revision, 'bytes': total, 'files': len(selected)}), flush=True)
    def fetch(f):
        name, expected = f['Path'], f['Sha256']
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == f['Size'] and digest(path) == expected:
            return {'path': name, 'cached': True, 'sha256': expected}
        url = f'https://modelscope.cn/api/v1/models/{model}/repo?' + urllib.parse.urlencode({'Revision': revision, 'FilePath': name})
        partial = path.with_suffix(path.suffix + '.partial')
        started = time.monotonic()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as response, partial.open('wb') as stream:
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                        stream.write(block)
                if partial.stat().st_size != f['Size'] or digest(partial) != expected:
                    raise ValueError('checksum mismatch: ' + name)
                partial.replace(path)
                result = {'path': name, 'bytes': f['Size'], 'sha256': expected, 'seconds': round(time.monotonic()-started, 1)}
                print(json.dumps(result), flush=True)
                return result
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipt = list(pool.map(fetch, selected))
    (out / 'source.json').write_text(json.dumps({'model': model, 'revision': revision, 'files': receipt}, indent=2))
    print('SDXL download verified', flush=True)

if __name__ == '__main__':
    main()
