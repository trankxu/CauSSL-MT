import os
import sys
# from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.models import DenseNet121,DenseNet161
from utils import losses, ramps
from utils.metrics import compute_AUCs
from utils.metric_logger import MetricLogger
from dataloaders import  dataset
from dataloaders.dataset import TwoStreamBatchSampler
from utils.util import get_timestamp
from validation import epochVal, epochVal_metrics, epochVal_metrics_test

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data1/bitgroup/trankxu/code/SRC-MT/data/skin/ISIC2018_Task3_Training_Input', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='/data1/bitgroup/trankxu/code/SRC-MT/data/skin/training.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='/data1/bitgroup/trankxu/code/SRC-MT/data/skin/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='/data1/bitgroup/trankxu/data/ISIC2019/filtered_2019.csv', help='testing set csv file')
#parser.add_argument('--root_path', type=str, default='/data1/bitgroup/trankxu/code/dataset/kvair/labelled_images', help='dataset root dir')
#parser.add_argument('--csv_file_train', type=str, default='/data1/bitgroup/trankxu/code/dataset/train.csv', help='training set csv file')
#parser.add_argument('--csv_file_val', type=str, default='/data1/bitgroup/trankxu/code/dataset/validation.csv', help='validation set csv file')
#parser.add_argument('--csv_file_test', type=str, default='/data1/bitgroup/trankxu/code/dataset/testing.csv', help='testing set csv file')
parser.add_argument('--exp', type=str,  default='Causalmodel', help='model_name')
parser.add_argument('--epochs', type=int,  default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default= 16, help='number of labeled data per batch')
parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model')
parser.add_argument('--labeled_num', type=int, default=1400, help='number of labeled')
parser.add_argument('--base_lr', type=float,  default=2e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1998, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### tune
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
# parser.add_argument('--resume', type=str,  default=None, help='GPU to use')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
### costs
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
parser.add_argument('--consistency_relation_weight', type=int,  default=1, help='consistency relation weight')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')
args = parser.parse_args()


train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
base_lr = args.base_lr
labeled_bs = args.labeled_bs * len(args.gpu.split(','))

if args.deterministic:
    cudnn.benchmark = False#

    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def correlation_penalty(causal, non_causal, lambda_corr= 0.01): #Causal && Non-causal: batch_size x 1024 x 7 x 7
    # Flatten the feature maps and compute their means and standard deviations
    causal = causal.view(causal.size(0), -1)
    non_causal = non_causal.view(non_causal.size(0), -1)
    causal_mean = torch.mean(causal, dim=0)
    non_causal_mean = torch.mean(non_causal, dim=0)
    causal_std = torch.std(causal, dim=0)
    non_causal_std = torch.std(non_causal, dim=0)

    # Compute the correlation coefficient
    cov = torch.mean((causal - causal_mean) * (non_causal - non_causal_mean), dim=0)
    
    corr = cov / (causal_std * non_causal_std)
    
    # Return a penalty term that increases with the absolute value of the correlation
    return lambda_corr * torch.abs(corr).sum()

class MINE(nn.Module):
    def __init__(self, input_size):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 1)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        joint = torch.cat([x, y], dim=1)
        marginal = torch.cat([x, torch.roll(y, shifts=1)], dim=1)
        t_statistic_joint = self.fc2(self.relu(self.fc1(joint)))
        t_statistic_marginal = self.fc2(self.relu(self.fc1(marginal)))
        return torch.abs(torch.mean(t_statistic_joint) - torch.log(torch.mean(torch.exp(t_statistic_marginal))))

def mutual_information_loss(mine_model, mask_part, non_mask_part):
    # Flatten the feature maps
    mask_part = mask_part.view(mask_part.size(0), -1)
    non_mask_part = non_mask_part.view(non_mask_part.size(0), -1)
    
    loss = mine_model(mask_part, non_mask_part)
    return loss

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = DenseNet121(out_size=dataset.N_CLASSES, mode=args.label_uncertainty, drop_rate=args.drop_rate)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        model = net.cuda()
        if ema:
            for param in model.parameters():#
                param.detach_()#
        return model
    
    ##########################Chuankai Add#####################################
    last_conv_features = None
    def Dense121_hook(module, input_, output):
        last_conv_features = output
    
    model = create_model()
    ema_model = create_model(ema=True)
    
    mine_model = MINE(100352).cuda()
    
    #optimizer = torch.optim.Adam([{'params':model.parameters(), 'lr':args.base_lr},{'params': mine_model.parameters(), 'lr':2e-4}], 
                                 #betas=(0.9, 0.999), weight_decay=5e-4)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, 
                                 betas=(0.9, 0.999), weight_decay=5e-4)
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found atW '{}'".format(args.resume)
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.global_step = checkpoint['global_step']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    weak_trans = transforms.Compose([transforms.Resize((224, 224)),
                        transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                        ])
                        
    strong_trans =  transforms.Compose([transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),     
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
            ])
            
    # dataset
    

    train_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                            csv_file=args.csv_file_train,
                                            transform=dataset.TransformTwice(weak_trans, weak_trans))

    val_dataset = dataset.CheXpertDataset(root_dir=args.root_path,
                                          csv_file=args.csv_file_val,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))
    test_dataset = dataset.CheXpertDataset(root_dir='/data1/bitgroup/trankxu/data/ISIC2019/ISIC_2019_Training_Input/',
                                          csv_file=args.csv_file_test,
                                          transform=transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              normalize,
                                          ]))

    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, 7000))# 1:5
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                                  num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)#, worker_init_fn=worker_init_fn)
    
    
    loss_fn = losses.cross_entropy_loss()
    #loss_mi = losses.RenyiMutualInformation(alpha=1.5, input_dim=50176, output_dim=50176, num_steps = 700, lr=0.01)
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')

    iter_num = args.global_step
    lr_ = base_lr
    model.train()
    
    ##########Chuankai Add
    best_AUROC_avg, best_AUROC_avg2, best_AUROC_avg3 = 0.0, 0.0, 0.0
    best_Accus_avg, best_Accus_avg2, best_Accus_avg3 = 0.0, 0.0, 0.0
    best_Senss_avg, best_Senss_avg2, best_Senss_avg3 = 0.0, 0.0, 0.0
    best_Specs_avg, best_Specs_avg2, best_Specs_avg3 = 0.0, 0.0, 0.0
    best_F1_avg, best_F1_avg2, best_F1_avg3 = 0.0, 0.0, 0.0
    
    #train
    for epoch in range(args.start_epoch, args.epochs):
    
        ##########日志记录显示
        meters_loss = MetricLogger(delimiter="  ")
        meters_loss_classification = MetricLogger(delimiter="  ")
        meters_loss_consistency = MetricLogger(delimiter="  ")
        meters_loss_consistency_relation = MetricLogger(delimiter="  ")
        meters_mask_loss = MetricLogger(delimiter="  ")
        meters_mi_loss = MetricLogger(delimiter="  ")
        meters_loss_consistency_cluster = MetricLogger(delimiter="  ")
        time1 = time.time()
        
        
       
        iter_max = len(train_dataloader)  # 训练集长度
        
        for i, (_,_, (image_batch, ema_image_batch), label_batch) in enumerate(train_dataloader):#items, index, image, torch.FloatTensor(label)
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()
            # unlabeled_image_batch = ema_image_batch[labeled_bs:]
            
            noise1 = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.1, 0.1)
            noise2 = torch.clamp(torch.randn_like(ema_image_batch) * 0.1, -0.1, 0.1)
            ema_inputs = ema_image_batch #+ noise2
            inputs = image_batch #+ noise1
            #print('inputs shape:',inputs.shape,'ema_inputs shape', ema_inputs.shape
             
            ################Chuankai change#####################
            
            non_mask_part,mask_part, non_mask_out, mask_out, activations = model(inputs)
            '''
            import scipy.io as io
            result1 = np.array(mask_part.view(64,-1).detach().cpu().numpy())
            np.savetxt('mask_part.txt',result1)
            result2 = np.array(non_mask_part.view(64,-1).detach().cpu().numpy())
            np.savetxt('non_mask_part.txt',result2)
            '''
            ###########End 
            
            with torch.no_grad():#已经detach了还有必要再no_grad()嘛
                ema_non_mask_part, ema_mask_part, ema_non_mask_out, ema_mask_out, ema_activations = ema_model(ema_inputs)
            
            #################Chuankai Change####################################
            ## calculate the loss
            from torch.distributions import uniform
            
            #kl_loss = losses.softmax_kl_loss
            #uniform_dist = uniform.Uniform(torch.zeros_like(non_mask_part), torch.ones_like(non_mask_part))
            #print(mask_part.shape)
            kl_loss = nn.KLDivLoss(reduction="mean", log_target=True)
            
            rand_var = 1/7 *torch.ones(non_mask_out.size()).cuda()
            
            '''
            for i in range(non_mask_part.size()[0]):
                min_val = non_mask_part[i].min().cuda()
                max_val = non_mask_part[i].max().cuda()
                rand_var[i] = torch.rand(non_mask_part.size()[1:]).cuda()*(max_val-min_val)
            '''    
            mask_loss =  loss_fn(non_mask_out,rand_var)
            #mask_loss = torch.sum(losses.relation_mse_loss(non_mask_out,rand_var)) / batch_size
            #print(mask_loss.shape)

            mi_loss = mutual_information_loss(mine_model, mask_part, non_mask_part) #+ mutual_information_loss(ema_mask_part, non_mask_part)#+ mutual_information_loss(mine_model, ema_mask_part,non_mask_part)
            
            #########End#################################
            
            loss_classification = loss_fn(mask_out[:labeled_bs], label_batch[:labeled_bs])
            
            
            loss = loss_classification + mask_loss #+ mi_loss
            
            ## MT loss (have no effect in the beginneing)
            if args.ema_consistency == 1:
                consistency_weight = get_current_consistency_weight(epoch)
                #print('outputs shape',outputs)
                #print('ema output shape', ema_output)
                consistency_dist = torch.sum(consistency_criterion(mask_out, ema_mask_out)) / batch_size #/ dataset.N_CLASSES
                #print('consistency distance shape', consistency_dist,consistency_weight)
                consistency_loss = consistency_weight * consistency_dist  

                #consistency_relation_dist = torch.sum(losses.relation_mse_loss_cam(activations, ema_activations, model, label_batch)) / batch_size
                
                consistency_relation_dist = torch.sum(losses.relation_mse_loss(activations, ema_activations)) / batch_size
                consistency_relation_loss = consistency_weight * consistency_relation_dist * args.consistency_relation_weight
                
                     
                
            else:
                consistency_loss = 0.0
                consistency_relation_loss = 0.0
                consistency_weight = 0.0
                consistency_dist = 0.0
             #+ consistency_loss

            if (epoch > 25) and (args.ema_consistency == 1):
                loss = loss_classification + consistency_loss# + consistency_relation_loss #+ consistency_cluster_relation_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # outputs_soft = F.softmax(outputs, dim=1)
            meters_loss.update(loss=loss)
            meters_loss_classification.update(loss=loss_classification)
            meters_loss_consistency.update(loss=consistency_loss)
            meters_loss_consistency_relation.update(loss=consistency_relation_loss)
            meters_mask_loss.update(loss=mask_loss)
            meters_mi_loss.update(loss=mi_loss)
           
            iter_num = iter_num + 1

            # write tensorboard
            if i % 100 == 0:
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_classification', loss_classification, iter_num)
                writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
                
                #logging.info("\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, classification loss: {:.6f}, consistency loss: {:.6f}, consistency relation loss: {:.6f}, consistency weight: {:.6f}, lr: {}"
                            #.format(epoch, i, iter_max, meters_loss.loss.avg, meters_loss_classification.loss.avg, meters_loss_consistency.loss.avg, meters_loss_consistency_relation.loss.avg, consistency_weight, optimizer.param_groups[0]['lr']))
                ########MT logging info
                logging.info("\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, classification loss: {:.6f}, consistency loss: {:.6f}, mask loss: {:.6f}, mi loss: {:.6f}, consistency weight: {:.6f}, lr: {}"
                            .format(epoch, i, iter_max, meters_loss.loss.avg, meters_loss_classification.loss.avg, meters_loss_consistency.loss.avg, meters_mask_loss.loss.avg,meters_mi_loss.loss.avg, consistency_weight, optimizer.param_groups[0]['lr']))

                image = inputs[-1, :, :]
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('raw/Image', grid_image, iter_num)

                image = ema_inputs[-1, :, :]
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('noise/Image', grid_image, iter_num)
                writer.add_histogram('non_mask_part', non_mask_part, iter_num)
                writer.add_histogram('mask_part', mask_part, iter_num)
        timestamp = get_timestamp()

        # validate student
        # 

        AUROCs, Accus, Senss, Specs, Pre, F1 = epochVal_metrics_test(model, val_dataloader)  
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        F1_avg = np.array(F1).mean()
        
        logging.info("\nVAL Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nVAL AUROC: {:6f}, VAL Accus: {:6f}, VAL Senss: {:6f}, VAL Specs: {:6f}, VAL F1: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(AUROCs)]))
        
        # test student
        AUROCs, Accus, Senss, Specs, Pre, F1 = epochVal_metrics_test(model, test_dataloader)
        
        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        F1_avg = np.array(F1).mean()
        
        logging.info("\nTEST Student: Epoch: {}, iteration: {}".format(epoch, i))
        logging.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
                    .format(AUROC_avg, Accus_avg, Senss_avg, Specs_avg, F1_avg))
        logging.info("AUROCs: " + " ".join(["{}:{:.6f}".format(dataset.CLASS_NAMES[i], v) for i,v in enumerate(AUROCs)]))
 
        if AUROC_avg > best_AUROC_avg:
            best_F1_avg = F1_avg
            best_AUROC_avg = AUROC_avg
            best_Accus_avg = Accus_avg
            best_Senss_avg = Senss_avg
            best_Specs_avg = Specs_avg
            epoch_auc = epoch
        
        if Accus_avg > best_Accus_avg2:
            best_F1_avg2 = F1_avg
            best_AUROC_avg2 = AUROC_avg
            best_Accus_avg2 = Accus_avg
            best_Senss_avg2 = Senss_avg
            best_Specs_avg2 = Specs_avg
            epoch_acc = epoch

        if F1_avg > best_F1_avg3:
            best_F1_avg3 = F1_avg
            best_AUROC_avg3 = AUROC_avg
            best_Accus_avg3 = Accus_avg
            best_Senss_avg3 = Senss_avg
            best_Specs_avg3 = Specs_avg
            epoch_f1 = epoch

        # save model
        '''
        save_mode_path = os.path.join(snapshot_path + 'checkpoint/', 'epoch_' + str(epoch+1) + '.pth')
        torch.save({    'epoch': epoch + 1,
                        'global_step': iter_num,
                        'state_dict': model.state_dict(),
                        'ema_state_dict': ema_model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'epochs'    : epoch,
                        # 'AUROC'     : AUROC_best,
                   }
                   , save_mode_path
        )
        logging.info("save model to {}".format(save_mode_path))
        '''
        # update learning rate
        lr_ = lr_ * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
    logging.info("\nBest TEST Student:")
    logging.info("\nAppear in Epoch: {}, TEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
                    .format(epoch_auc, best_AUROC_avg, best_Accus_avg, best_Senss_avg, best_Specs_avg, best_F1_avg))
    logging.info("\nAppear in Epoch: {}, TEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
                    .format(epoch_acc, best_AUROC_avg2, best_Accus_avg2, best_Senss_avg2, best_Specs_avg2, best_F1_avg2))
    logging.info("\nAppear in Epoch: {}, TEST AUROC: {:6f}, TEST Accus: {:6f}, TEST Senss: {:6f}, TEST Specs: {:6f}, TEST F1: {:6f}"
                    .format(epoch_f1, best_AUROC_avg3, best_Accus_avg3, best_Senss_avg3, best_Specs_avg3, best_F1_avg3))
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(iter_num+1)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
