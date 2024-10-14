import warnings

warnings.filterwarnings("ignore")
import numpy as np
from torchvision import transforms
import os

import torch
import argparse
from dataset.dataset_ec import ECdataset

from sklearn.metrics import f1_score
from time import time
from utils.load_weights import *
from utils.sam import SAM

from models.FSFNet import FSFNet
from dataset.randomaug import RandAugment

# log
import datetime
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ec', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=128, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.000004, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=300, help='Total training epochs.')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    return parser.parse_args()


import random
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run_training():
    args = parse_args()
    set_random_seed(42)

    log_txt_path = 'log/' + time_str + 'set' + str(args.dataset) + '-log.txt'


    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])


    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        RandAugment(1,5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1)),
    ])

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    num_classes = 4
    datapath = '/home/mafuyan/Pictures/Multimediate2024/ec_train_val'
    train_dataset = ECdataset(datapath, train=True, transform=data_transforms)
    val_dataset = ECdataset(datapath, train=False, transform=data_transforms_val)
    model = FSFNet(img_size=112, num_classes=num_classes, type=args.modeltype)


    val_num = val_dataset.__len__()
    print('Train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    model = model.cuda()

    print("batch_size:", args.batch_size)

    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        checkpoint = checkpoint["model_state_dict"]
        model = load_pretrained_weights(model, checkpoint)

    params = model.parameters()
    if args.optimizer == 'adamw':
        # base_optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        # base_optimizer = torch.optim.Adam(params, args.lr, weight_decay=1e-4)
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        # base_optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=1e-4)
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")
    # print(optimizer)
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False,)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)
    CE_criterion = torch.nn.CrossEntropyLoss()


    best_acc = 0
    for i in range(1, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        start_time = time()
        model.train()
        for batch_i, (imgs, targets) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            outputs, features = model(imgs)
            targets = targets.cuda()

            CE_loss = CE_criterion(outputs, targets)
            loss = CE_loss
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            outputs, features = model(imgs)
            CE_loss = CE_criterion(outputs, targets)

            loss = CE_loss
            loss.backward() # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)

            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss / iter_cnt
        elapsed = (time() - start_time) / 60

        print('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Loss: %.3f LR:%.6f' %
              (i, elapsed, train_acc, train_loss, optimizer.param_groups[0]["lr"]))

        with open(log_txt_path, 'a') as f:
            f.write('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Loss: %.3f LR:%.6f' %
              (i, elapsed, train_acc, train_loss, optimizer.param_groups[0]["lr"]) + '\n')

        scheduler.step()

        pre_labels = []
        gt_labels = []
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for batch_i, (imgs, targets) in enumerate(val_loader):
                outputs, features = model(imgs.cuda())
                targets = targets.cuda()

                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()

            val_loss = val_loss / iter_cnt
            val_acc = bingo_cnt.float() / float(val_num)
            val_acc = np.around(val_acc.numpy(), 4)
            f1 = f1_score(pre_labels, gt_labels, average='macro')
            total_socre = 0.67 * f1 + 0.33 * val_acc

            print("[Epoch %d] Validation accuracy:%.4f, Loss:%.3f, f1 %4f, score %4f" % (
            i, val_acc, val_loss, f1, total_socre))

            with open(log_txt_path, 'a') as f:
                f.write("[Epoch %d] Validation accuracy:%.4f" % (
                    i, val_acc) + '\n')

            if val_acc > best_acc:
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('./checkpoint', "best.pth"))
                print('Model saved.')

                # log
                with open(log_txt_path, 'a') as f:
                    f.write("Model_saved." + '\n')

                best_acc = val_acc
                print("best_acc:" + str(best_acc))

    print("best_acc:" + str(best_acc))


if __name__ == "__main__":
    run_training()