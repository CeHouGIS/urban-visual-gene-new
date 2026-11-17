"""Review actual spatial selections and real four-direction photos (CPU GIS/I/O)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import shapely

from common import load_config, write_json, now
from data import decode_rgb


def create_report(config):
    root=Path(config['root'])
    out=root/'reports/spatial_sampling'
    out.mkdir(parents=True,exist_ok=True)
    grids=gpd.read_parquet(root/'data/selected_grids.parquet')
    points=pd.read_parquet(root/'data/sampled_points.parquet')
    # Audit the selected geospatial records, independently of quota allocation.
    expected={s:config['sampling'][f'{s}_cells'] for s in ['train','validation','test']}
    if grids['split'].value_counts().to_dict()!=expected or grids['grid_id'].duplicated().any():
        raise AssertionError('Actual selected grid counts or split isolation failed')
    if points['pano'].duplicated().any() or not (points.groupby('grid_id').size()==5).all():
        raise AssertionError('Actual selected points violate five unique points per grid')
    if points.groupby('grid_id')['split'].nunique().max()!=1:
        raise AssertionError('Point records cross grid splits')
    owner=grids.set_index('grid_id').loc[points['grid_id']]
    if not shapely.intersects(shapely.points(points['x'],points['y']),owner.geometry.to_numpy()).all():
        raise AssertionError('Selected coordinates lie outside their clipped GHSL grid')
    if not (points['split'].to_numpy()==owner['split'].to_numpy()).all():
        raise AssertionError('Selected point and grid split labels disagree')
    write_json(out/'spatial_integrity.json',{'status':'passed','exact_grid_counts':expected,
        'unique_panorama_points':len(points),'one_split_per_global_grid':True,
        'five_points_per_grid':True,'all_points_inside_clipped_GHSL_grid':True,'updated_utc':now()})
    training=grids[grids['split']=='train']
    distribution=pd.read_csv(root/'data/distribution_by_continent.csv')
    order=sorted(training['continent'].unique())
    counts=distribution.pivot(index='continent',columns='split',values='photos').reindex(order).fillna(0)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    counts[['train','validation','test']].plot.barh(stacked=True,ax=ax,color=['#4083b7','#e3a54d','#6aa379'])
    ax.set_xlabel('Real photographs'); ax.set_ylabel('')
    ax.set_title('GHSL grid sample: capacity-aware continent / country / city balance')
    fig.savefig(out/'continent_distribution.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
    examples=[]
    for ax,continent in zip(axes.flat,order):
        subset=training[training['continent']==continent].sort_values(['country','ucdb_id','grid_id'])
        grid=subset.iloc[len(subset)//2]
        selected=points[points['grid_id']==grid['grid_id']].reset_index(drop=True)
        gpd.GeoSeries([grid.geometry],crs=grids.crs).plot(ax=ax,facecolor='#dfebdc',edgecolor='#507b49')
        ax.scatter(selected['x'],selected['y'],s=50,color='#c84639',zorder=3)
        for index,point in selected.iterrows():
            ax.annotate(str(index+1),(point['x'],point['y']),xytext=(5,5),textcoords='offset points')
        side=config['sampling']['grid_size_m']
        left,bottom=grid['gx']*side,grid['gy']*side
        ax.set_xlim(left,left+side);ax.set_ylim(bottom,bottom+side);ax.set_aspect('equal')
        ax.set_xticks([left,left+side],['0','1000 m']);ax.set_yticks([bottom,bottom+side],['0','1000 m'])
        ax.set_title(f"{continent}\n{grid['city']}, {grid['country']}",fontsize=10)
        paths=[]
        for point in selected.to_dict('records'):
            directory=Path(config['sampling']['image_root'])/point['source_country']/point['source_city']/point['pano'][0]/point['pano'][1]
            paths.extend(str(directory/f"{point['pano']}_{h}.jpg") for h in config['sampling']['headings'])
        tile=160; sheet=Image.new('RGB',(tile*4,(tile+23)*5+30),'white');draw=ImageDraw.Draw(sheet)
        draw.text((5,5),f"{continent} | {grid['grid_id']} | 5 points x 4 headings",fill='black')
        with ThreadPoolExecutor(max_workers=8) as pool:
            images=list(pool.map(lambda p:decode_rgb(p,tile),paths))
        for index,rgb in enumerate(images):
            row,col=divmod(index,4); x,y=col*tile,30+row*(tile+23)
            draw.text((x+3,y+4),f"Point {row+1} | {config['sampling']['headings'][col]} deg",fill='black')
            sheet.paste(Image.fromarray(rgb),(x,y+23))
        name=continent.lower().replace(' ','_')+'.jpg'
        sheet.save(out/name,quality=90)
        examples.append({'continent':continent,'grid_id':grid['grid_id'],'GHSL_city':grid['city'],
                         'country':grid['country'],'contact_sheet':str(out/name),'photos':paths})
    fig.suptitle('One selected 1 km projected grid per continent; shaded = GHSL intersection',fontsize=13)
    fig.savefig(out/'six_example_grids.png',dpi=180);plt.close(fig)
    summary={'training_photos':len(training)*20,'training_grids':len(training),
             'training_GHSL_cities':int(training['ucdb_id'].nunique()),'training_countries':int(training['country'].nunique()),
             'split_grids':grids['split'].value_counts().to_dict(),
             'continent_photos':counts.astype(int).to_dict(),'examples':examples,'updated_utc':now()}
    write_json(out/'summary.json',summary)
    lines=['# GHSL 格网抽样检查','',f"训练集：{len(training):,} 个格网，{len(training)*20:,} 张真实街景。",'',
           '按洲→国家→城市分配配额，但各洲的既有合格图片容量差别很大，因此实际数量仍有差异。',
           '同一个格网的 5 个点、每点 4 张图始终属于一个集合。', '',
           '![洲分布](continent_distribution.png)','', '![各洲格网实例](six_example_grids.png)','',
           '| 大洲 | 训练图 | 验证图 | 测试图 |','|---|---:|---:|---:|']
    for continent,row in counts.iterrows():
        lines.append(f"| {continent} | {int(row['train']):,} | {int(row['validation']):,} | {int(row['test']):,} |")
    for example in examples:
        lines+=['',f"## {example['continent']}：{example['GHSL_city']}，{example['country']}", '',
                f"格网 `{example['grid_id']}`，每行一个位置，每列一个朝向。",'',
                f"![5 点四朝向]({Path(example['contact_sheet']).name})"]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();create_report(load_config(args.config))
