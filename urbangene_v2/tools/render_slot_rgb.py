#!/usr/bin/env python3
"""Replay one saved continuous-slot program as RGB without reading any photograph."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    import numpy as np
    from PIL import Image
    import torch
    from diffusers import AutoencoderKL
    from urbangene_v2.common import writable
    from urbangene_v2.composition import SlotProgram
    from urbangene_v2.discovery import DiscoveryModel
    from urbangene_v2.rgb_probe import SlotRGBBridge, freeze_module, render_slot_program

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', required=True)
    parser.add_argument('--bridge', required=True)
    parser.add_argument('--discovery', required=True)
    parser.add_argument('--vae', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = writable(args.output)
    if output.exists():
        raise FileExistsError(output)
    saved = torch.load(args.bridge, map_location='cpu', weights_only=False)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in saved['protocol']['existing_allocation']:
        raise ValueError('Select one of the two originally allocated GPU UUIDs')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('One GPU required, CPU inference disabled')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.2)
    checkpoint = torch.load(args.discovery, map_location='cpu', weights_only=False)
    if checkpoint['step'] != saved['protocol']['checkpoint_step']:
        raise ValueError('Discovery checkpoint and adapter provenance do not match')
    model = freeze_module(DiscoveryModel(**checkpoint['config']['model'])).cuda()
    model.load_state_dict(checkpoint['model'], strict=True)
    final = model.decoder.network[-1]
    bridge = SlotRGBBridge(final.weight[:-1], final.bias[:-1], width=saved['width']).cuda().eval()
    bridge.load_state_dict(saved['model'], strict=True)
    vae = freeze_module(AutoencoderKL.from_pretrained(
        args.vae, local_files_only=True, use_safetensors=True, low_cpu_mem_usage=False)).cuda()
    program = SlotProgram.load(args.program, 'cuda:0')
    with torch.inference_mode():
        rgb = render_slot_program(program, model.decoder, bridge, vae)[0]
    array = (rgb.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output)
    print(output)


if __name__ == '__main__':
    main()
