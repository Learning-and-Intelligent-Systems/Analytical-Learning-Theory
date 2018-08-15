'''
Created on Aug 15, 2018

@author: vermavik
'''
from __future__ import division

import os, sys, shutil, time, random
import argparse
from distutils.dir_util import copy_tree
from shutil import rmtree
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models

import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
 import _pickle as pickle
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from load_data  import *
from helpers import *
from plots import *

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))
print (model_names)

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnext29_8_64', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--singlecutout', action='store_true', default=False,
                    help='whether to use singlecutout')
parser.add_argument('--dualcutout', action='store_true', default=False,
                    help='whether to use dualcutout')
parser.add_argument('--cutsize', type=int, default=16, help='cutout size.')
parser.add_argument('--dropout', action='store_true', default=False,
                    help='whether to use dropout or not in final layer')
#parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.05, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--alpha', type=float, default=0.01, help='the coefficient that controls the difference between the outputs from two cutouts')
parser.add_argument('--data_aug', type=int, default=0)
parser.add_argument('--add_name', type=str, default='')
#parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--decay', type=float, default=0.0000, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=1000, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--job_id', type=str, default='')
parser.add_argument('--temp_dir', type = str, default = '/Tmp/vermavik/',
                        help='folder on local node where data is stored temporarily')
parser.add_argument('--home_dir', type = str, default = '/data/milatmp1/vermavik/',
                        help='file where results are to be written')



args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

out_str = str(args)
print(out_str)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True


def experiment_name(arch='',
                    epochs=400,
                    dropout=True,
                    batch_size=64,
                    lr=0.01,
                    momentum=0.5,
                    alpha= 0.01,
                    decay=0.0005,
                    data_aug=1,
                    dualcutout= False,
                    singlecutout= False,
                    cutsize = 16,
                    manualSeed=None,
                    job_id=None,
                    add_name=''):

    exp_name= str(arch)
    exp_name += '_epochs_'+str(epochs)
    if dropout:
        exp_name+='_dropout_'+'true'
    else:
        exp_name+='_dropout_'+'False'
    if dualcutout:
        exp_name+='_dualcutout_'+'true'
        exp_name +='_cut_size_'+str(cutsize)
    elif singlecutout:
        exp_name+='_singlecutout_'+'true'
        exp_name +='_cut_size_'+str(cutsize)
    else:
        exp_name+='_nocutout_'+'true'

    exp_name +='_batch_size_'+str(batch_size)
    exp_name += '_lr_'+str(lr)
    exp_name += '_momentum_'+str(momentum)
    exp_name += '_alpha_'+str(alpha)
    exp_name +='_decay_'+str(decay)
    exp_name += '_data_aug_'+str(data_aug)
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if manualSeed!=None:
        exp_name += '_manuael_seed_'+str(manualSeed)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)

    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# train function (forward, backward, update)
def train(train_loader, model, criterion, cutout,  optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if  args.dualcutout == True or args.singlecutout == True :
            cutout1 = cutout.apply(input)
            cutout2 = cutout.apply(input)
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()
                cutout1 = cutout1.cuda()
                cutout2 = cutout2.cuda()

            input_var = Variable(input)
            cutout1_var = Variable(cutout1)
            cutout2_var = Variable(cutout2)
            target_var = Variable(target)

            # compute output
            output1 = model(cutout1_var)
            if args.dualcutout:
                output2 = model(cutout2_var)
            if args.dualcutout:
                loss = (criterion(output1, target_var)+criterion(output2, target_var))*0.5 + args.alpha*F.mse_loss(output1, output2)
            else:
                loss = criterion(output1, target_var)

            total_loss = loss
        # measure accuracy and record loss

        else:
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()

            input_var = Variable(input)
            target_var = Variable(target)

            # compute output
            output1 = model(input_var)
            loss = criterion(output1, target_var)

            total_loss = loss



        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if args.dualcutout:
            prec1, prec5 = accuracy((output1.data+output2.data)*0.5, target, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output1.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)


    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg

best_acc = 0
def main():

    ### transfer data from source to current node#####
    print ("Copying the dataset to the current node's  dir...")

    tmp = args.temp_dir
    home = args.home_dir


    dataset=args.dataset
    data_source_dir = os.path.join(home,'data',dataset)
    if not os.path.exists(data_source_dir):
        os.makedirs(data_source_dir)
    data_target_dir = os.path.join(tmp,'data',dataset)
    copy_tree(data_source_dir, data_target_dir)

    ### set up the experiment directories########
    exp_name=experiment_name(arch=args.arch,
                    epochs=args.epochs,
                    dropout=args.dropout,
                    batch_size=args.batch_size,
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    alpha = args.alpha,
                    decay= args.decay,
                    data_aug=args.data_aug,
                    dualcutout=args.dualcutout,
                    singlecutout = args.singlecutout,
                    cutsize = args.cutsize,
                    manualSeed=args.manualSeed,
                    job_id=args.job_id,
                    add_name=args.add_name)
    temp_model_dir = os.path.join(tmp,'experiments/DualCutout/'+dataset+'/model/'+ exp_name)
    temp_result_dir = os.path.join(tmp, 'experiments/DualCutout/'+dataset+'/results/'+ exp_name)
    model_dir = os.path.join(home, 'experiments/DualCutout/'+dataset+'/model/'+ exp_name)
    result_dir = os.path.join(home, 'experiments/DualCutout/'+dataset+'/results/'+ exp_name)


    if not os.path.exists(temp_model_dir):
        os.makedirs(temp_model_dir)

    if not os.path.exists(temp_result_dir):
        os.makedirs(temp_result_dir)

    copy_script_to_folder(os.path.abspath(__file__), temp_result_dir)

    result_png_path = os.path.join(temp_result_dir, 'results.png')


    global best_acc

    log = open(os.path.join(temp_result_dir, 'log.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(temp_result_dir), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)


    train_loader, test_loader,num_classes=load_data(args.data_aug, args.batch_size,args.workers,args.dataset, data_target_dir)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer

    net = models.__dict__[args.arch](num_classes,args.dropout)
    print_log("=> network :\n {}".format(net), log)

    #net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)


    cutout = Cutout(1, args.cutsize)
    if args.use_cuda:
        net.cuda()
    criterion.cuda()

    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        validate(test_loader, net, criterion, log)
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    # Main loop
    train_loss = []
    train_acc=[]
    test_loss=[]
    test_acc=[]
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        tr_acc, tr_los = train(train_loader, net, criterion, cutout, optimizer, epoch, log)

        # evaluate on validation set
        val_acc,   val_los   = validate(test_loader, net, criterion, log)
        train_loss.append(tr_los)
        train_acc.append(tr_acc)
        test_loss.append(val_los)
        test_acc.append(val_acc)
        dummy = recorder.update(epoch, tr_los, tr_acc, val_los, val_acc)

        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc

        save_checkpoint({
          'epoch': epoch + 1,
          'arch': args.arch,
          'state_dict': net.state_dict(),
          'recorder': recorder,
          'optimizer' : optimizer.state_dict(),
        }, is_best, temp_model_dir, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(result_png_path)

    train_log = OrderedDict()
    train_log['train_loss'] = train_loss
    train_log['train_acc']=train_acc
    train_log['test_loss']=test_loss
    train_log['test_acc']=test_acc

    pickle.dump(train_log, open( os.path.join(temp_result_dir,'log.pkl'), 'wb'))
    plotting(temp_result_dir)

    copy_tree(temp_model_dir, model_dir)
    copy_tree(temp_result_dir, result_dir)

    rmtree(temp_model_dir)
    rmtree(temp_result_dir)

    log.close()


if __name__ == '__main__':
    main()
