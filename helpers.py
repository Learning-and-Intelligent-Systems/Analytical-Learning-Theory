'''
Created on 16 Nov 2017

@author: vermav1
'''
from time import gmtime, strftime
import torch
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def apply(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def experiment_name(cod=True,
                    cod_trainable=False,
                    aux_nets=2,
                    opt='sgd',
                    epochs=400,
                    batch_size=64,
                    test_batch_size=1000,
                    lr=0.01,
                    momentum=0.5, 
                    data_aug=1,
                    manualSeed=None,
                    job_id=None,
                    add_name=''):
    if cod:
        exp_name = 'cod_true'
        if cod_trainable:
            exp_name+='_trainable_true'
        else:
            exp_name+='_trainable_false'
    else:
        exp_name = 'cod_false'
    exp_name+='_auxnets_'+str(aux_nets)
    exp_name+='_opt_'+str(opt)
    exp_name+='_epochs_'+str(epochs)
    exp_name +='_batch_size_'+str(batch_size)
    exp_name+='_test_batch_size_'+str(test_batch_size)
    exp_name += '_lr_'+str(lr)
    exp_name += '_momentum_'+str(momentum)
    exp_name += '_data_aug_'+str(data_aug)
    if manualSeed!=None:
        exp_name += '_manuael_seed_'+str(manualSeed)
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)
    
    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name


def experiment_name_non_mnist(arch='',
                    aux_nets=2,
                    epochs=400,
                    dropout=True,
                    batch_size=64,
                    lr=0.01,
                    momentum=0.5, 
                    data_aug=1,
                    manualSeed=None,
                    job_id=None,
                    add_name=''):
    
    exp_name= str(arch)
    exp_name+='_auxnets_'+str(aux_nets)
    exp_name += '_epochs_'+str(epochs)
    if dropout:
        exp_name+='_dropout_'+'true'
    else:
        exp_name+='_dropout_'+'False'
    exp_name +='_batch_size_'+str(batch_size)
    exp_name += '_lr_'+str(lr)
    exp_name += '_momentum_'+str(momentum)
    exp_name += '_data_aug_'+str(data_aug)
    if manualSeed!=None:
        exp_name += '_manuael_seed_'+str(manualSeed)
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)
    
    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name

def copy_script_to_folder(caller_path, folder):
    script_filename = caller_path.split('/')[-1]
    script_relative_path = os.path.join(folder, script_filename)
    # Copying script
    shutil.copy(caller_path, script_relative_path)

def cyclic_lr(initial_lr,step,total_steps,num_cycles):
    factor=np.ceil(float(total_steps)/num_cycles)
    theta=np.pi*np.mod(step-1,factor)/factor
    return (initial_lr/2)*(np.cos(theta)+1)

if __name__ == '__main__':
    lr_list=[]
    for i in xrange(1000):
        lr=cyclic_lr(0.1,i+1,1100,3)
        lr_list.append(lr)
    plt.plot(np.asarray(lr_list))
    plt.show()
        