"""GPU-generated all-cap reconstructions, real errors, and portable code examples."""
import numpy as np
from PIL import Image,ImageDraw
import torch
from common import write_json
from data import image_tensor
from codec import pack

def to_pil(value):
 return Image.fromarray(np.rint(value.detach().float().clamp(0,1).cpu().permute(1,2,0).numpy()*255).astype(np.uint8))

def sheet(rows,destination,side=256):
 canvas=Image.new('RGB',(side*len(rows[0]),(side+28)*len(rows)),'white');draw=ImageDraw.Draw(canvas)
 for i,row in enumerate(rows):
  for j,(label,value) in enumerate(row):
   draw.text((j*side+4,i*(side+28)+6),label,fill='black');canvas.paste(to_pil(value).resize((side,side)),(j*side,i*(side+28)+28))
 canvas.save(destination)

@torch.inference_mode()
def preview(model,pixels,config,output,label):
 prior_mode,prior_rate=model.training,model.rate_index;model.eval()
 try:
  ids=np.linspace(0,len(pixels)-1,min(config['training']['preview_count'],len(pixels)),dtype=int)
  x=image_tensor(pixels,ids);rows=[[('Original',value)] for value in x];metadata={'step':model.step,'indices':ids.tolist(),'rates':{}};codes={}
  out=output/'previews';out.mkdir(exist_ok=True)
  for rate,cap in enumerate(config['model']['budgets']):
   model.rate_index=rate
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):result=model(x)
   count,payload=model.actual_complexity(result)
   for i in range(len(x)):rows[i].append((f'Cap {cap} / used {int(count[i])}',result['reconstruction'][i]))
   if rate==0:min_prediction=result['reconstruction']
   metadata['rates'][str(cap)]={'active_genes':count.cpu().tolist(),'actual_encoded_bytes':payload.cpu().tolist(),'scale_caps':{s:model.quota(s) for s in model.scales}}
   for s,a in result['activations'].items():
    q=np.rint(a[:2].float().cpu().numpy()/config['model']['expression_max']*255).astype(np.uint8)
    codes[f'cap{cap}_{s}_uint8']=q
    for i in range(min(2,len(x))):(out/f'{label}.cap{cap}.image{i}.{s}.bin').write_bytes(pack(a[i].float().cpu().numpy(),config['model']['expression_max']))
  for i in range(len(x)):rows[i].append(('Cap8 abs RGB error x3',(min_prediction[i]-x[i]).abs()*3))
  sheet(rows,out/f'{label}.png',x.shape[-1]);write_json(out/f'{label}.json',metadata)
  np.savez_compressed(out/f'{label}.codes.npz',image_indices=ids[:2],**codes)
 finally:model.train(prior_mode);model.rate_index=prior_rate
