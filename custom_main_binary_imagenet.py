import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import models
from torch.autograd import Variable
from utils import *
from datetime import datetime
from ast import literal_eval
import dataset
import torchvision.transforms as transforms

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

# Logging
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder (named by datetime)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--seed', default=1234, type=int,
                    help='random seed')
# Model
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--infl_ratio', default=1, type=float,
                    help='infl ratio of channels')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of classes')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
# Training
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--cache_size', default=50000, type=int,
                    help='cache size for data loader')
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--batch_size_test', default=32, type=int,
                    help='mini-batch size for testing (default: 32)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--distrloss', default=0, type=float,
                    help='weight of distrloss')
parser.add_argument('--distr_epoch', default=100, type=int,
                    help='epochs to add distrloss')


def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    random.seed(args.seed)
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'infl_ratio': args.infl_ratio, 'num_classes': args.num_classes}

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)
    logging.info("model structure: %s", model)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    transform = getattr(model, 'input_transform', {})
    regime = getattr(model, 'regime', {})
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_loader = dataset.SeqImageNetLoader_org('val', batch_size=args.batch_size_test, num_workers=args.workers, cuda=True, remainder=True,
                       transform=transform['eval'], cache=args.cache_size, shuffle=False)

    if args.evaluate:
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0, args.distrloss)
        logging.info('\n Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
        return

    train_loader = dataset.SeqImageNetLoader_org('train', batch_size=args.batch_size, num_workers=args.workers, cuda=True, remainder=False,
                       transform=transform['train'], cache=args.cache_size)

    optimizer = torch.optim.Adam(model.parameters()) # Ruizhou: here we pass model parameters iter to optimizer
    logging.info('transform: %s', transform)
    ### logging.info('training regime: %s', regime)


    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])

        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer, args.distrloss)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch, args.distrloss)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        # save model checkpoint every few epochs
        if epoch % 1 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'config': model_config,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'regime': regime,
                'parameters': list(model.parameters()),
            }, is_best, path=save_path)

        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, args_distrloss=0.5):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distr_losses1 = AverageMeter()
    distr_losses2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=False) ###not training)
        target_var = Variable(target)

        # compute output
        output, distr_loss1, distr_loss2 = model(input_var)
        distr_loss1 = distr_loss1.mean()
        distr_loss2 = distr_loss2.mean()
        loss = criterion(output, target_var)
        # remove distrloss after 20 epochs
        if epoch < args.distr_epoch and args_distrloss > 1e-4:
            loss = loss + (distr_loss1 + distr_loss2) * args_distrloss

        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        distr_losses1.update(distr_loss1.data.item(), inputs.size(0))
        distr_losses2.update(distr_loss2.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            for p in list(model.parameters()):
                if hasattr(p,'org'): 
                    p.data.copy_(p.org)
            optimizer.step() 
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1)) 

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Distrloss1 {distrloss1.val:.8f}({distrloss1.avg:.8f})\t'
                         'Distrloss2 {distrloss2.val:.8f}({distrloss2.avg:.8f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, 
                             distrloss1=distr_losses1, distrloss2=distr_losses2, 
                             top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer, args_distrloss=0.5):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer, args_distrloss=args_distrloss)


def validate(data_loader, model, criterion, epoch, args_distrloss=0.5):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None, args_distrloss=0)


if __name__ == '__main__':
    main()
