"""Behavioral GPU gates including replay, support, parent freeze and DDP math."""
import argparse
import copy
import io
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from common import load_config, now, write_json
from model import SequentialGene, objective, pyramid, tiles, untile, up


def synchronize_buffers(model):
    maximum=torch.zeros((),device='cuda')
    for value in model.buffers():
        reference=value.clone()
        dist.broadcast(reference,0)
        maximum=torch.maximum(maximum,(value.float()-reference.float()).abs().max())
    dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
    assert float(maximum)==0, float(maximum)


def run(config):
    torch.manual_seed(819)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    x=torch.rand(2,3,256,256,device='cuda')
    for g in [1,2,4,8]:
        torch.testing.assert_close(untile(tiles(x,g),g),x,rtol=0,atol=0)
    target=pyramid(x)
    replay=target[0]
    for i in range(1,4):
        replay=up(replay)+(target[i]-up(target[i-1]))
    torch.testing.assert_close(replay,x,rtol=0,atol=1e-6)
    model=SequentialGene(config['model']).cuda()
    model.set_phase(0)
    model.train()
    result=model(x[:1])
    assert int(model.levels[0].book.updates)==1
    model.set_phase(3)
    frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('levels.3.')}
    order=[]
    hooks=[level.register_forward_hook(lambda module, args, output, i=i: order.append(i)) for i,level in enumerate(model.levels)]
    result=model(x[:1])
    for h in hooks:
        h.remove()
    assert order==[0,1,2,3]
    for i in range(4):
        torch.testing.assert_close(result['parents'][i]+result['corrections'][i],result['outputs'][i],rtol=0,atol=0)
        if i:
            torch.testing.assert_close(result['parents'][i],up(result['outputs'][i-1]),rtol=0,atol=0)
    loss,_,_=objective(result,3,config['model']['beta'])
    loss.backward()
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    opt.step()
    for k,v in frozen.items():
        torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0,msg=lambda msg:k+': '+msg)
    assert all(p.grad is None for level in model.levels[:3] for p in level.parameters())
    assert any(p.grad is not None for p in model.levels[3].encoder.parameters())
    synchronize_buffers(model)
    model.eval()
    before={k:v.clone() for k,v in model.named_buffers()}
    with torch.no_grad():
        result=model(x,last=3)
        for k,v in before.items():
            torch.testing.assert_close(dict(model.named_buffers())[k],v,rtol=0,atol=0)
        stream=io.BytesIO()
        torch.save([v.to(torch.int16).cpu() for v in result['indices']],stream)
        stream.seek(0)
        ids=[v.long().cuda() for v in torch.load(stream,weights_only=True)]
        decoded=model.decode_indices(ids)
        for a,b in zip(decoded,result['outputs']):
            torch.testing.assert_close(a,b,rtol=0,atol=2e-6)
        changed=[v.clone() for v in ids]
        changed[3][:,0,0]=(changed[3][:,0,0]+1)%256
        alter=model.decode_indices(changed)[-1]
        difference=(alter-decoded[-1]).abs()
        outside=difference.clone()
        outside[:,:,:32,:32]=0
        assert float(outside.max())<1e-6
        assert float(difference.max())>1e-7
    # DDP gradients vs full-batch reference, with frozen codebooks so the two
    # computations use the same mapping and do not change it between forwards.
    model.zero_grad(set_to_none=True)
    reference=copy.deepcopy(model)
    ddp=DDP(model,device_ids=[torch.cuda.current_device()],broadcast_buffers=False)
    rank=dist.get_rank()
    output=ddp(x[rank:rank+1])
    objective(output,3,config['model']['beta'])[0].backward()
    reference.zero_grad(set_to_none=True)
    objective(reference(x),3,config['model']['beta'])[0].backward()
    error=0.
    for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad,q.grad,rtol=3e-3,atol=3e-5,msg=lambda msg:name+': '+msg)
            error=max(error,float((p.grad-q.grad).abs().max()))
    # The model must reject CPU neural execution before any computation.
    try:
        model(torch.zeros(1,3,256,256))
        raise AssertionError('CPU guard absent')
    except RuntimeError as e:
        assert 'GPU-only' in str(e)
    if rank==0:
        write_json(Path(config['root'])/'reports/gpu_tests.json',dict(status='passed',updated_utc=now(),
            checks=['tile replay','target pyramid replay','ordered forward','actual predicted parent',
                    'additive correction','frozen parent weights and EMA','evaluation EMA immutability',
                    'index serialization/decode replay','last-cell spatial support','distributed EMA agreement',
                    'DDP gradient equivalence','CPU model guard'],ddp_max_gradient_absolute_error=error))
    dist.barrier()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    try:
        run(load_config(args.config))
    finally:
        dist.destroy_process_group()
