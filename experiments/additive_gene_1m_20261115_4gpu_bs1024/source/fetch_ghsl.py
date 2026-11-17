"""Resume an official GHSL reference download using bounded HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import shutil
import time
import zipfile

import requests

from common import now, write_json

URL='https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_UCDB_GLOBE_R2024A/GHS_UCDB_GLOBE_R2024A/V1-2/GHS_UCDB_GLOBE_R2024A_V1_2.zip'
TOTAL=276662440


def download(target):
    target=Path(target)
    directory=target.parent
    directory.mkdir(parents=True,exist_ok=True)
    status=directory/'download_progress.json'
    if target.exists() and target.stat().st_size==TOTAL and zipfile.is_zipfile(target):
        write_json(status,{'status':'complete','bytes':TOTAL,'source':URL,'updated_utc':now()})
        return target
    parts=directory/'download_parts'
    parts.mkdir(exist_ok=True)
    prefix=parts/'prefix.bin'
    if target.exists():
        if prefix.exists():
            raise ValueError('Both prefix and an incomplete target exist; inspect before resuming')
        target.rename(prefix)
    initial=prefix.stat().st_size if prefix.exists() else 0
    if initial>=TOTAL:
        raise ValueError('Unexpected reference file size')
    chunk=math.ceil((TOTAL-initial)/16)
    ranges=[(i,start,min(start+chunk,TOTAL)-1) for i,start in enumerate(range(initial,TOTAL,chunk))]
    def fetch(item):
        i,start,end=item
        path=parts/f'part_{i:02d}.bin'
        length=end-start+1
        for attempt in range(3):
            have=path.stat().st_size if path.exists() else 0
            if have==length:
                return
            if have>length:
                raise ValueError('Reference part exceeds its assigned range')
            try:
                with requests.get(URL,headers={'Range':f'bytes={start+have}-{end}'},stream=True,timeout=(20,60)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start+have}-{end}/{TOTAL}':
                        raise ValueError('Server did not honor the requested byte range')
                    with path.open('ab') as stream:
                        for data in response.iter_content(256*1024):
                            stream.write(data)
                if path.stat().st_size!=length:
                    raise ValueError('Reference range transfer incomplete')
                return
            except Exception:
                if attempt==2:
                    raise
                time.sleep(2)
    write_json(status,{'status':'downloading','bytes':initial,'total':TOTAL,'connections':16,'source':URL,'updated_utc':now()})
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(fetch,item) for item in ranges]
        while not all(f.done() for f in futures):
            received=initial+sum(p.stat().st_size for p in parts.glob('part_*.bin'))
            write_json(status,{'status':'downloading','bytes':received,'total':TOTAL,'connections':16,'source':URL,'updated_utc':now()})
            time.sleep(3)
        for future in futures:
            future.result()
    assembled=directory/'reference_assembled.partial'
    with assembled.open('wb') as stream:
        if prefix.exists():
            with prefix.open('rb') as source:
                shutil.copyfileobj(source,stream,1024*1024)
        for i,start,end in ranges:
            with (parts/f'part_{i:02d}.bin').open('rb') as source:
                shutil.copyfileobj(source,stream,1024*1024)
    if assembled.stat().st_size!=TOTAL or not zipfile.is_zipfile(assembled):
        raise ValueError('Assembled official reference has an unexpected size or container format')
    assembled.replace(target)
    write_json(status,{'status':'complete','bytes':TOTAL,'source':URL,'updated_utc':now()})
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    download(args.output)
