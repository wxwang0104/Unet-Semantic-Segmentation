import argparse
import random

import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import sys
sys.path.append('./..')
from dataset.dataset import SemanticSegmentationDataset
from models.unet import UNet
from train import Trainer
from submit import Submitor


def print_to_log(description, value, f):
    print(description + ': ', value)
    print(description + ': ', value, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to run segmentation models')
    parser.add_argument('--not_debug', help='exit from debug mode', action='store_false', dest='debug', default=False)
    parser.add_argument('--use_gpu', help='Debug the models', action='store_true', default=False)
    parser.add_argument('--batch_size', help='desired batch size for training', action='store', type=int, dest='batch_size', default=1)
    parser.add_argument('--num_classes', help='number of classes for prediction', action='store', type=int, dest='num_classes', default=2)
    parser.add_argument('--output_dir', help='path to saving outputs', action='store', dest='output_dir', default='results')
    parser.add_argument('--model', help='models to train on', action='store', dest='model', default='unet')
    parser.add_argument('--learning_rate', help='starting learning rate', action='store', type=float, dest='learning_rate', default=0.001)
    parser.add_argument('--optimizer', help='adam or sgd optimizer', action='store', dest='optimizer', default='sgd')
    parser.add_argument('--random_seed', help='seed for random initialization', action='store', type=int, dest='seed', default=100)
    parser.add_argument('--load_model', help='load models from file', action='store_true', default=True)
    parser.add_argument('--predict', help='only predict', action='store_true', default=True)
    parser.add_argument('--unet_batch_norm', help='to choose whether use batch normalization for unet', action='store_true', default=False)
    parser.add_argument('--unet_use_dropout', help='use unet dropout', action='store_true', default=False)
    parser.add_argument('--unet_dropout_rate', help='to set the dropout rate for unet', action='store', type=float, default=0.5)
    parser.add_argument('--unet_channels', help='the number of unet first conv channels', action='store', type=int, default=32)
    parser.add_argument('--print_every', help='print loss every print_every steps', action='store', type=int, default=10)
    parser.add_argument('--save_model_every', help='save models every save_model_every steps', action='store', type=int, default=500)
    parser.add_argument('--crop_size', help='crop image to this size', action='store', type=int, default=224)
    parser.add_argument('--pretrained', help='load pretrained models when doing transfer learning', action='store_true', default=True)
    parser.add_argument('--num_epochs', help='total number of epochs for training', action='store', type=int, default=100000)
    parser.add_argument('--is_validation', help='whether or not calculate validation when training', action='store_true', default=False)
    parser.add_argument('--validation_every', help='calculate validation loss every validation_every steps', action='store', type=int, default=1)
    parser.add_argument('--lr_decay_every', help='learning rate decay every lr_decay_every steps', action='store', type=int, default=10000)
    parser.add_argument('--lr_decay_ratio', help='learning rate decay ratio', action='store', type=float, default=0.5)
    parser.add_argument('--is_auto_adjust_rate', help='if using auto adjust learning rate', action='store_true', default=True)
    parser.add_argument('--lr_adjust_every', help='number of iterations to check learning rate', action='store', type=int, default=1000)
    parser.add_argument('--train_dir', help='training file location', action='store', default='data/train')
    parser.add_argument('--validation_dir', help='validation file location', action='store', default='data/train')
    parser.add_argument('--test_dir', help='testing file location', action='store', default='data/test')
    parser.add_argument('--out_dir', help='prediction output file location', action='store', default='prediction')
    parser.add_argument('--train_set', help='the training image set file location', action='store',
                        choices=[
                            'train1_ids_gray2_500'],
                        default='train1_ids_gray2_500')
    parser.add_argument('--valid_set', help='the validation image set file location', action='store',
                        choices=[
                            'valid1_ids_gray2_43'],
                        default='valid1_ids_gray2_43')
    parser.add_argument('--test_set', help='the testing image set file location', action='store',
                        choices=[
                            'test1_ids_gray2_53'],
                        default='test1_ids_gray2_53')
    parser.add_argument('--initial_checkpoint', help='the initialization checkpoint', action='store', default='00000059_model.pth')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(args.output_dir)
    log_dir = os.path.join(args.output_dir, 'log.txt')
    log_file = open(log_dir, 'a')

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        print_to_log('random_seed', args.seed, log_file)

    if args.debug:
        print_every = 1
        save_model_every = 10
        image_size = 224
        pretrained = True
        batch_size = 1
        learning_rate = 0.01
        n_epochs = 1000
        is_validation = False
        validation_every = 10
        unet_batch_norm = True
        unet_use_dropout = False
        unet_dropout_rate = None
        predict = False
        lr_decay_every = 100
        lr_decay_ratio = 0.5
        is_auto_adjust_rate = True
        lr_adjust_every = 1000
        load_model = False

    else:
        print_every = args.print_every
        save_model_every = args.save_model_every
        image_size = args.crop_size
        pretrained = args.pretrained
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        n_epochs = args.num_epochs
        is_validation = args.is_validation
        validation_every = args.validation_every
        unet_batch_norm = args.unet_batch_norm
        unet_use_dropout = args.unet_use_dropout
        unet_dropout_rate = args.unet_dropout_rate if unet_use_dropout else None
        predict = args.predict
        lr_decay_every = args.lr_decay_every
        lr_decay_ratio = args.lr_decay_ratio
        is_auto_adjust_rate = args.is_auto_adjust_rate
        lr_adjust_every = args.lr_adjust_every
        load_model = args.load_model

    print_to_log('debug', args.debug, log_file)
    print_to_log('batch size', batch_size, log_file)
    print_to_log('num_classes', args.num_classes, log_file)
    print_to_log('output_dir', args.output_dir, log_file)
    print_to_log('models', args.model, log_file)
    print_to_log('learning_rate', args.learning_rate, log_file)
    print_to_log('optimizer', args.optimizer, log_file)
    print_to_log('load_model', load_model, log_file)
    print_to_log('unet unet_batch_norm', unet_batch_norm, log_file)
    print_to_log('unet_use_dropout', unet_use_dropout, log_file)
    print_to_log('unet_dropout_rate', unet_dropout_rate, log_file)
    print_to_log('unet_channels', args.unet_channels, log_file)
    print_to_log('print_every', args.print_every, log_file)
    print_to_log('save_model_every', args.save_model_every, log_file)
    print_to_log('validation_every', validation_every, log_file)
    print_to_log('crop_size', args.crop_size, log_file)
    print_to_log('predict', predict, log_file)
    print_to_log('lr_decay_every', lr_decay_every, log_file)
    print_to_log('lr_decay_ratio', lr_decay_ratio, log_file)
    print_to_log('is_auto_adjust_rate', is_auto_adjust_rate, log_file)
    print_to_log('lr_adjust_every', lr_adjust_every, log_file)
    print_to_log('training set', args.train_set, log_file)

    if args.model == 'unet':
        model = UNet(3, n_classes=args.num_classes, first_conv_channels=args.unet_channels, batch_norm=unet_batch_norm, dropout_rate=unet_dropout_rate)
    else:
        raise Exception('Unknown models')
    if load_model:
        model_path = os.path.join(args.output_dir, 'checkpoint', args.initial_checkpoint)
        print_to_log('loading models from file', args.initial_checkpoint, log_file)
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        # model.load_state_dict(torch.load(model_path))
        # torch.load(model_path)
        print_to_log('models loaded!', '', log_file)

    print("Running models: " + args.model)
    cuda = torch.cuda.is_available() and args.use_gpu
    if cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    print_to_log('gpu', cuda, log_file)
    log_file.close()

    if not predict:
        print("training on train set")
        train_set = SemanticSegmentationDataset(args.train_dir,
                                                os.path.join('image_sets/', args.train_set),
                                                image_size, mode='train')
        val_set = SemanticSegmentationDataset(args.validation_dir,
                                              os.path.join('image_sets/', args.valid_set),
                                              image_size, mode='valid')

        loss = torch.nn.CrossEntropyLoss()
        train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size, drop_last=False)
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise Exception('Unknown optimizer')

        if load_model:
            optimizer_path = os.path.join(args.output_dir, 'checkpoint', args.initial_checkpoint.replace('_model.pth', '_optimizer.pth'))
            checkpoint = torch.load(optimizer_path)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate  # load all except learning rate

        trainer = Trainer(cuda=cuda,
                          model=model,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          loss=loss,
                          optimizer=optimizer,
                          n_epochs=n_epochs,
                          n_save=save_model_every,
                          n_print=print_every,
                          learning_rate=learning_rate,
                          is_validation=is_validation,
                          lr_decay_every=lr_decay_every,
                          lr_decay_ratio=lr_decay_ratio,
                          is_auto_adjust_rate=is_auto_adjust_rate,
                          lr_adjust_every=lr_adjust_every,
                          out_dir=args.output_dir)
        trainer.train()

    else:
        print("predicting on test set")
        test_set = SemanticSegmentationDataset(args.test_dir,
                                               os.path.join('image_sets/', args.test_set),
                                               crop_size=image_size, mode='test')
        test_loader = DataLoader(dataset=test_set, batch_size=1)

        mysubmitor = Submitor(model, test_loader, output_dir=args.output_dir, cuda=cuda, threshold=20, saveseg=True)
        mysubmitor.generate_submission_file('prediction')
