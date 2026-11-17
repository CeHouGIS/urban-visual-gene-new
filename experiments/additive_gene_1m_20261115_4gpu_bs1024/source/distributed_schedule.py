"""Sample-based scheduling preserves the original shuffle and training budget."""
import math
import numpy as np
from data import epoch_permutation


def sample_ids(offset,count,batch_size,seed):
    if offset<0 or batch_size<=0:
        raise ValueError('Invalid sample offset or batch size')
    parts=[]
    while batch_size:
        epoch,start=divmod(offset,count)
        take=min(batch_size,count-start)
        parts.append(epoch_permutation(count,seed,epoch)[start:start+take])
        offset+=take
        batch_size-=take
    return np.concatenate(parts)


def rank_ids(offset,count,batch_size,seed,rank,world):
    if batch_size%world or not 0<=rank<world:
        raise ValueError('Global batch must split evenly across valid ranks')
    return np.split(sample_ids(offset,count,batch_size,seed),world)[rank]


def learning_rate(images_seen,settings):
    old_batch=settings['previous_effective_batch_size']
    old_step=images_seen/old_batch
    warmup=settings['warmup_updates']
    if old_step<warmup:
        return settings['learning_rate']*(old_step+1)/warmup
    fraction=(old_step-warmup)/max(1,settings['previous_max_updates']-warmup-1)
    fraction=min(1.0,max(0.0,fraction))
    return settings['minimum_learning_rate']+(settings['learning_rate']-settings['minimum_learning_rate'])*0.5*(1+math.cos(math.pi*fraction))


def next_batch(images_seen,settings):
    return min(settings['effective_batch_size'],settings['max_training_images']-images_seen)
