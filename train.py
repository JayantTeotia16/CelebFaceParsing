import os
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score
#from torchvision.models import shufflenet_v2_x0_5

from model import Mobi, Mobief
from utils.models import _load_model_weights, model_dict
from utils.dataGen import make_data_generator
from utils.losses import get_loss
from utils.metrics import dice_coeff, MetricTracker
from utils.lr_scheduler import LR_Scheduler

class Trainer:
    def __init__(self, config, custom_losses = None):
        self.f_name = config['f_name']
        self.checkpoint_dir = os.path.join('./', self.f_name, 'checkpoints')
        self.logs_dir = os.path.join('./', self.f_name, 'logs')

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.isdir(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        self.config = config
        self.batch_size = self.config['batch_size']
        self.model_name = self.config['model_name']
        self.model_path = self.config.get('model_path', None)
        #self.enc1 = shufflenet_v2_x0_5(pretrained=False)
        self.model = Mobi()
        if self.model_path:
            self.model = _load_model_weights(self.model, self.model_path)
        
        self.train_df, self.val_df = get_train_val_dfs(self.config)
        self.train_datagen = make_data_generator(self.config, self.train_df, stage = 'train')
        self.val_datagen = make_data_generator(self.config, self.val_df, stage = 'val')
        self.epochs = self.config['training']['epochs']
        self.lr = self.config['training']['lr']
        self.loss = get_loss(self.config['training'].get('loss'),
                             self.config['training'].get('loss_weights'),
                             custom_losses)
        self.metrics = dice_coeff
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
        else:
            self.gpu_count = 0

        self.train_writer = SummaryWriter(os.path.join(self.logs_dir, 'runs', self.f_name, 'training'))
        self.val_writer = SummaryWriter(os.path.join(self.logs_dir, 'runs', self.f_name, 'val'))

        self.initialize_model()

    def initialize_model(self):
        if self.gpu_available:
            self.model = self.model.cuda()
            if self.gpu_count > 1:
                self.model.DataParallel(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr,
                                         momentum = 0.9, weight_decay = 1e-4, nesterov=True)
        self.lr_scheduler = LR_Scheduler('poly', self.lr, self.epochs + 1, len(self.train_datagen))
        

    def run(self):
        best_metric = 0
        print(self.model)
        print(sum(p.numel() for p in self.model.parameters()))
        self.test()
        #print("Encoder",sum(p.numel() for p in self.enc1.parameters()))
        # for epoch in range(1, self.epochs + 1):
        #     print('Epoch {}/{}'.format(epoch, self.epochs))
        #     print('-' * 10)
        #     self.train(epoch, best_metric)
        #       metric_v = self.val(epoch)
        #     is_best_metric = metric_v > best_metric
        #     best_metric = max(metric_v, best_metric)
        #     self.save_checkpoint({
        #         'epoch': epoch,
        #         'state_dict': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
        #         'best_metric': best_metric,
        #         'optimizer': self.optimizer.state_dict()
        #     }, is_best_metric)

    def train(self, epoch, best_metric):
        Losses = MetricTracker()
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_datagen, desc="training", ascii=True, ncols=60)):
            if torch.cuda.is_available():
                data = batch['image'].cuda()
                target = batch['mask'].cuda().long()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            Losses.update(loss.item(), data.size(0))
            self.lr_scheduler(self.optimizer, idx, epoch, best_metric)

        info = {
            "Loss": Losses.avg,
        }
        for tag, value in info.items():
            self.train_writer.add_scalar(tag, value, epoch)
        
        print('Train Loss: {:.6f}'.format(
                Losses.avg
                ))

        return None
    
    def val(self, epoch):
        self.model.eval()
        torch.cuda.empty_cache()
        val_Metric = MetricTracker()
        area_intersect_all = np.zeros(19)
        area_union_all = np.zeros(19)
        f1_scores = []
        f1_scores_per_image = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.val_datagen, desc="val", ascii=True, ncols=60)):
                if torch.cuda.is_available():
                    data = batch['image'].cuda()
                    target = batch['mask'].cuda().float()
                
                logits = self.model(data)
                outputs = torch.argmax(logits, dim=1).float()
                outputs = outputs.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                for cls_idx in range(19):
                    area_intersect = np.sum((outputs == target) * (outputs == cls_idx))

                    area_pred_label = np.sum(outputs == cls_idx)
                    area_gt_label = np.sum(target == cls_idx)
                    area_union = area_pred_label + area_gt_label - area_intersect

                    area_intersect_all[cls_idx] += area_intersect
                    area_union_all[cls_idx] += area_union
                #val_Metric.update(self.metrics(outputs, target), outputs.size(0))
                true_labels = target.flatten()  # Flatten the ground truth labels
                predicted_labels = outputs.flatten()
                for class_idx in range(19):
                    # Get binary labels for the current class (One-vs-Rest)
                    true_binary = (true_labels == class_idx).astype(int)
                    predicted_binary = (predicted_labels == class_idx).astype(int)
                    
                    # Calculate F1 score for the current class
                    f1 = f1_score(true_binary, predicted_binary, average='binary', zero_division=1)
                    f1_scores.append(f1)
                
                # Store the F1 scores for the current image
                f1_scores_per_image.append(f1_scores)
            iou_all = area_intersect_all / area_union_all * 100.0
            miou = iou_all.mean()
            print(miou,"MIOU")
            print(np.mean(f1_scores_per_image),"F1")
        info = {
            "Dice": miou#val_Metric.avg
        }
        for tag, value in info.items():
            self.val_writer.add_scalar(tag, value, epoch)
        
        print('Val Dice: {:.6f}'.format(
                miou#val_Metric.avg
                ))

        return miou#val_Metric.avg
    
    def test(self):
        self.model.eval()
        torch.cuda.empty_cache()
        val_Metric = MetricTracker()
        area_intersect_all = np.zeros(19)
        area_union_all = np.zeros(19)
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.val_datagen, desc="val", ascii=True, ncols=60)):
                if torch.cuda.is_available():
                    data = batch['image'].cuda()                
                logits = self.model(data)
                outputs = torch.argmax(logits, dim=1).float().squeeze()
                outputs = outputs.detach().cpu().numpy()
                print(batch['path'][-1], outputs.shape)
                print(batch['path'][-1].split("/")[-1][:-3])
                result = Image.fromarray(outputs.astype(np.uint8))
                result.save('./output/'+batch['path'][-1].split("/")[-1][:-3]+'png')
                
        return None
    
    def save_checkpoint(self, state, is_best):
        filename = os.path.join(self.checkpoint_dir, self.f_name + '_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.checkpoint_dir, self.f_name + '_model_best.pth.tar'))




def get_train_val_dfs(config):
    train_df = pd.read_csv(config['training_data_csv'])
    val_df = pd.read_csv(config['validation_data_csv'])
    return train_df, val_df