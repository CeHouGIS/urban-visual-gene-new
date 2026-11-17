"""Reformat existing saved preview PNGs on CPU; no new model inference."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'runs/multiscale_residual_ista/previews'
output=ROOT/'reports/display'
output.mkdir(exist_ok=True)
font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
small=ImageFont.truetype(str(font_path),14) if font_path.exists() else font
reconstruction=Image.open(source/'test_best.png').convert('RGB')
pyramid=Image.open(source/'test_best.pyramid.png').convert('RGB')
side=256
header=34

def tile(image,row,column):
    return image.crop((column*side,row*280+24,(column+1)*side,row*280+280))

rgb=Image.new('RGB',(3*side,2*(side+header)), 'white')
draw=ImageDraw.Draw(rgb)
labels=['Original photo','Reconstruction','Absolute RGB error x3']
for row in range(2):
    for col,label in enumerate(labels):
        x,y=col*side,row*(side+header)
        draw.text((x+5,y+8),label,font=font,fill='black')
        rgb.paste(tile(reconstruction,row,col),(x,y+header))
rgb.save(output/'original_reconstruction_error.png')

scales=Image.new('RGB',(4*side,3*(side+header)),'white')
draw=ImageDraw.Draw(scales)
for scale_idx,scale in enumerate(['Coarse','Medium','Fine']):
    for sample in range(2):
        for kind,label in enumerate(['target','prediction']):
            col=sample*2+kind;x,y=col*side,scale_idx*(side+header)
            draw.text((x+5,y+8),f'Photo {sample+1}: {scale} {label}',font=small,fill='black')
            scales.paste(tile(pyramid,sample,2+scale_idx*2+kind),(x,y+header))
scales.save(output/'residual_targets_predictions.png')
metadata={'source_pngs':[str(source/'test_best.png'),str(source/'test_best.pyramid.png')],
          'sample_indices':[0,199],'model_checkpoint_step':6000,
          'new_model_inference':False,'operation':'lossless crop and rearrangement of saved 8-bit preview tiles; new headers',
          'rgb_error_display':'original saved abs(pred-target)*3, clipping for display',
          'residual_display':'original saved 0.5+2*residual for medium/fine; coarse in RGB',
          'interpretation':'auxiliary RGB residual heads; these are not additive RGB contributions to the final decoder'}
(output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(output/'original_reconstruction_error.png')
print(output/'residual_targets_predictions.png')
