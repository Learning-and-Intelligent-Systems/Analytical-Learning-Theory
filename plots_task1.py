'''
Created on 19 Oct 2017

@author: vermav1
'''
import argparse
import cPickle as pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
sns.set(color_codes=True)

plot_from_index=-10000


def plotting(exp_dir):
    # Load the training log dictionary:
    train_dict = pickle.load(open(os.path.join(exp_dir, 'log.pkl'), 'rb'))
    
    ###########################################################
    ### Make the vanilla train and test loss per epoch plot ###
    ###########################################################
   
    plt.plot(np.asarray(train_dict['train_loss']), label='train_loss')
        
    #plt.ylim(0,2000)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_loss.png' ))
    plt.clf()
    
    
    
    plt.plot(np.asarray(train_dict['test_loss']), label='test_loss')
       
    #plt.ylim(0,100)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'test_loss.png' ))
    plt.clf()
    
    plt.plot(np.asarray(train_dict['train_acc']), label='train_acc')
       
    #plt.ylim(0,100)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'train_acc.png' ))
    plt.clf()
    
    
    plt.plot(np.asarray(train_dict['test_acc']), label='test_acc')
       
    #plt.ylim(0,100)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'test_acc.png' ))
    plt.clf()
    
    plt.plot(np.asarray(train_dict['reg1']), label='reg1')
       
    #plt.ylim(0,100)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'reg1.png' ))
    plt.clf()
    
    
    plt.plot(np.asarray(train_dict['reg2']), label='reg2')
       
    #plt.ylim(0,100)
    plt.xlabel('evaluation step')
    plt.ylabel('metrics')
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(exp_dir, 'reg2.png' ))
    plt.clf()
    
       
if __name__ == '__main__':
    plotting('~/experiments/DARC/cifar10/results/epochs_300_dropout_true_batch_size_64_lr_0.05_momentum_0.9_alpha1_0.0_alpha2_0.0_alpha3_0.0_data_aug_0_manuael_seed_2526_job_id_1')
    #plotting_separate_theta('model', 'temp.pkl',3)