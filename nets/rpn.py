"""
先验框,建议框,图中左下角部分
"""


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox


#-----------------------------------------#
#   用于对建议框解码并进行非极大抑制
#-----------------------------------------#
class ProposalCreator():
    def __init__(
        self,
        mode,
        nms_iou             = 0.7,
        n_train_pre_nms     = 12000,
        n_train_post_nms    = 600,      # 前600个得分最高的建议框
        n_test_pre_nms      = 3000,
        n_test_post_nms     = 300,      # 前300个得分最高的建议框
        min_size            = 16

    ):
        #-----------------------------------#
        #   设置预测还是训练
        #-----------------------------------#
        self.mode               = mode
        #-----------------------------------#
        #   建议框非极大抑制的iou大小
        #-----------------------------------#
        self.nms_iou            = nms_iou
        #-----------------------------------#
        #   训练用到的建议框数量
        #-----------------------------------#
        self.n_train_pre_nms    = n_train_pre_nms
        self.n_train_post_nms   = n_train_post_nms  # 前600个得分最高的建议框
        #-----------------------------------#
        #   预测用到的建议框数量
        #-----------------------------------#
        self.n_test_pre_nms     = n_test_pre_nms
        self.n_test_post_nms    = n_test_post_nms   # 前300个得分最高的建议框
        self.min_size           = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        """

        Args:
            loc (Tensor): 先验框调整参数        [12996, 4]
            score (Tensor): 先验框得分          [12996]
            anchor (NdArray): 先验框            [12996, 4]
            img_size (Tensor): 图片大小 [H, W]  [600, 600]
            scale (float, optional): min_size的缩放系数. Defaults to 1..

        Returns:
            Tensor: 非极大抑制后的建议框 [300, 4]
        """
        if self.mode == "training":
            n_pre_nms   = self.n_train_pre_nms
            n_post_nms  = self.n_train_post_nms
        else:
            n_pre_nms   = self.n_test_pre_nms   # 3000
            n_post_nms  = self.n_test_post_nms  # 300

        #-----------------------------------#
        #   将先验框转换成tensor
        #-----------------------------------#
        anchor = torch.from_numpy(anchor).type_as(loc)  # [12996, 4]
        #-----------------------------------#
        #   将RPN网络预测结果转化成尚未筛选的建议框
        #   get xyxy
        #-----------------------------------#
        roi = loc2bbox(anchor, loc)                     # [12996, 4]
        #-----------------------------------#
        #   防止建议框超出图像边缘
        #-----------------------------------#
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])

        #-----------------------------------#
        #   建议框的宽高的最小值不可以小于16
        #-----------------------------------#
        min_size    = self.min_size * scale
        # [keep_len]
        keep        = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        #-----------------------------------#
        #   将对应的建议框保留下来
        #-----------------------------------#
        roi         = roi[keep, :]  # [keep_len, 4]
        score       = score[keep]   # [keep_len]

        #-----------------------------------#
        #   根据得分进行排序，取出建议框
        #-----------------------------------#
        order       = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order   = order[:n_pre_nms] # [3000]  max 3000
        roi     = roi[order, :] # [3000, 4]
        score   = score[order]  # [3000]

        #-----------------------------------#
        #   对建议框进行非极大抑制
        #   使用官方的非极大抑制会快非常多
        #-----------------------------------#
        keep    = nms(roi, score, self.nms_iou) # [keep_len]
        if len(keep) < n_post_nms:  # 如果keep的长度小于300,则在keep中随机选择一些id拼接原来的keep
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep        = torch.cat([keep, keep[index_extra]])
        keep    = keep[:n_post_nms] # [300]
        roi     = roi[keep]         # [300, 4]
        return roi

#-----------------------------------------#
#   生成先验框,获得建议框,图中左下部分
#-----------------------------------------#
class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        in_channels     = 512,
        mid_channels    = 512,
        ratios          = [0.5, 1, 2],
        anchor_scales   = [8, 16, 32],
        feat_stride     = 16,
        mode            = "training",
    ):
        super(RegionProposalNetwork, self).__init__()
        #-----------------------------------------#
        #   生成基础先验框，shape为[9, 4]
        #-----------------------------------------#
        self.anchor_base    = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)
        n_anchor            = self.anchor_base.shape[0]

        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合 左侧第一个卷积
        #-----------------------------------------#
        self.conv1  = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体       上面输出为18的卷积, 18=9x2,代表是否有物体(1个代表为背景概率,另一个代表为物体概率)
        #-----------------------------------------#
        self.score  = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        #-----------------------------------------#
        #   回归预测对先验框进行调整            下面输出为36的卷积, 36=9x4,代表调整框的参数,
        #-----------------------------------------#
        self.loc    = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        #-----------------------------------------#
        #   特征点间距步长
        #-----------------------------------------#
        self.feat_stride    = feat_stride
        #-----------------------------------------#
        #   用于对建议框解码并进行非极大抑制
        #-----------------------------------------#
        self.proposal_layer = ProposalCreator(mode)

        #--------------------------------------#
        #   对FPN的网络部分进行权值初始化
        #--------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """
        x: base_feature共享特征层 [B, 1024, 38, 38]
        img_size: 输入图片大小    [H, W] [600, 600]
        """
        n, _, h, w = x.shape
        #-----------------------------------------#
        #   先进行一个3x3Conv，可理解为特征整合
        #   [B, 1024, 38, 38] -> [B, 512, 38, 38]
        #-----------------------------------------#
        x = F.relu(self.conv1(x))
        #-----------------------------------------#
        #   1x1Conv回归预测对先验框进行调整
        #   [B, 512, 38, 38] -> [B, 36, 38, 38] -> [B, 38, 38, 36] -> [B, 38*38*9, 4]   38*38*9 = 12996
        #-----------------------------------------#
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        #-----------------------------------------#
        #   1x1Conv分类预测先验框内部是否包含物体
        #   [B, 512, 38, 38] -> [B, 18, 38, 38] -> [B, 38, 38, 18] -> [B, 38*38*9, 2]   38*38*9 = 12996
        #-----------------------------------------#
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        #--------------------------------------------------------------------------------------#
        #   获得先验框得分
        #   进行softmax概率计算，每个先验框只有两个判别结果  softmax变化到0~1之间,和为1
        #   rpn_scores: [B, 38*38*9, 2]  [:, :, 1] 取最后一维的最后一个
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        #--------------------------------------------------------------------------------------#
        rpn_softmax_scores  = F.softmax(rpn_scores, dim=-1)             # [B, 38*38*9, 2] -> [B, 38*38*9, 2]
        rpn_fg_scores       = rpn_softmax_scores[:, :, 1].contiguous()  # [B, 38*38*9, 2] get [B, 38*38*9]
        # rpn_fg_scores     = rpn_fg_scores.view(n, -1)                 # [B, 38*38*9] -> [B, 38*38*9] 无用

        #------------------------------------------------------------------------------------------------#
        #   生成先验框，此时获得的anchor是布满网格点的
        #   当输入图片为600x600x3的时候，公用特征层的shape就是38x38x1024，anchor的shape为 [12996, 4]
        #------------------------------------------------------------------------------------------------#
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)

        # 建议框,建议框索引
        rois        = list()
        roi_indices = list()
        for i in range(n):  # 循环每张图片的结果
            #-----------------------------------------#
            #   用于对建议框解码并进行非极大抑制
            #-----------------------------------------#
            roi         = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale = scale)   # [300, 4]
            batch_index = i * torch.ones((len(roi),))   # 为每个batch的roi设置相同的batch, [0 * 300, 1 * 300...]
            rois.append(roi.unsqueeze(0))               # [300, 4] -> [1, 300, 4]
            roi_indices.append(batch_index.unsqueeze(0))
        # 列表变为矩阵
        rois        = torch.cat(rois, dim=0).type_as(x)                         # [B, 300, 4]
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)                  # [B, 300]
        anchor      = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)# [12996, 4] -> [1, 12996, 4]

        # 先验框调整参数,先验框得分(是否包含物体),建议框,建议框索引,先验框
        # rpn_locs:     [B, 12996, 4]
        # rpn_scores:   [B, 12996, 2]
        # rois:         [B, 300, 4]
        # roi_indices:  [B, 300]
        # anchor:       [1, 12996, 4]
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
