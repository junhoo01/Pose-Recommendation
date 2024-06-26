# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints



class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion_class = nn.BCELoss()
        self.criterion_reg = nn.MSELoss(reduction='sum')
 
    def forward(self, output, target):
        #output.size = [128,17,3]
        #output_reg.size=[128,17,2]   [x,y]
        #output_class.size=[128,17,1]
        #target.size=[128,17,2] 
        reg_pred = output[:,:,:2]
        class_pred = output[:,:,2]
        reg_gt = target[:,:,:2]
        class_gt = target[:,:,2]
        loss_reg = 0

        #binary classification loss(binary cross entropy)
        loss_class = self.criterion_class(class_pred, class_gt)

        #class 0을 예측했다면 계산되지 않도록 pred, gt의 key_points를 0,0으로 바꿈
        # but exp.size=[128,17], key_points=[128,17,2]라 안될 가능성 존재
        mask = torch.stack((class_pred, class_pred), dim=2)
        key_points_pred = reg_pred.masked_fill(mask < 0.5, 0)
        key_points_gt = reg_gt.masked_fill(mask < 0.5, 0)
        
        #regression loss(MSELoss)
        #batch 별 class가 1인 joints 수를 세서 batch 별로 나눠줘야 함
        batch_size = output.size(0)
        for idx in range(batch_size):
            pred = key_points_pred[idx]
            gt = key_points_gt[idx]
            num_joints = 1
            for idx2 in range(17):
                if class_gt[idx, idx2] == 1: 
                    num_joints += 1
            loss_reg += 0.5 * self.criterion_reg(pred, gt) / num_joints
        return loss_class + loss_reg / batch_size



    
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, A, B):
        # 유클리드 거리 계산
        distance = torch.sqrt(torch.sum((A - B) ** 2))
        return distance