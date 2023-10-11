import torch.nn as nn

from nets.classifier import Resnet50RoIHead, VGG16RoIHead
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork
from nets.vgg16 import decom_vgg16

#--------------------------------------------------------------#
#   1. backbone提取出共享特征层
#   2. 生成建议框并用建议框进行筛选
#   3. 使用筛选的框截取共享特征层
#   4. 将截取后的结果进行分类预测和是否包含物体
#--------------------------------------------------------------#
class FasterRCNN(nn.Module):
    def __init__(self,  num_classes,
                    mode = "training",
                    feat_stride = 16,
                    anchor_scales = [8, 16, 32],
                    ratios = [0.5, 1, 2],
                    backbone = 'vgg',
                    pretrained = False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        #---------------------------------#
        #   一共存在两个主干
        #   vgg和resnet50
        #---------------------------------#
        if backbone == 'vgg':
            self.extractor, classifier = decom_vgg16(pretrained)
            #---------------------------------#
            #   构建建议框网络
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                in_channels     = 512,
                mid_channels    = 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建分类器网络
            #---------------------------------#
            self.head = VGG16RoIHead(
                n_class         = num_classes + 1,  # 分类数+是否包含物体
                roi_size        = 7,                # roi图片大小
                spatial_scale   = 1,
                classifier      = classifier
            )
        elif backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            #---------------------------------#
            #   构建建议框网络
            #   生成先验框,获得建议框,图中左下部分
            #---------------------------------#
            self.rpn = RegionProposalNetwork(
                in_channels     = 1024,
                mid_channels    = 512,
                ratios          = ratios,
                anchor_scales   = anchor_scales,
                feat_stride     = self.feat_stride,
                mode            = mode
            )
            #---------------------------------#
            #   构建分类器网络
            #---------------------------------#
            self.head = Resnet50RoIHead(
                n_class         = num_classes + 1,  # 分类数+是否包含物体
                roi_size        = 14,               # roi图片大小
                spatial_scale   = 1,
                classifier      = classifier
            )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            #---------------------------------#
            #   计算输入图片的大小 [H, W]
            #---------------------------------#
            img_size        = x.shape[2:]
            #---------------------------------#
            #   利用主干网络提取特征
            #   [B, 3, 600, 600] -> [B, 1024, 38, 38]
            #---------------------------------#
            base_feature    = self.extractor(x)

            #---------------------------------#
            #   获得建议框
            #   [B, 1024, 38, 38] -> rois:        [B, 300, 4]
            #                        roi_indices: [B, 300]
            #---------------------------------#
            _, _, rois, roi_indices, _  = self.rpn(base_feature, img_size, scale)
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #   rois:        [B, 300, 4] -> roi_cls_locs: [B, 300, (num_classes+1)*4]  针对每个类别都预测一个框的调整值,和yolo只预测1个框和这个框的种类不同
            #   roi_indices: [B, 300]       roi_scores:   [B, 300, num_classes+1]
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head(base_feature, rois, roi_indices, img_size)

            # roi_cls_locs: 建议框的调整参数  [B, 300, (num_classes+1)*4]  针对每个类别都预测一个框的调整值,和yolo只预测1个框和这个框的种类不同
            # roi_scores:   建议框的种类得分  [B, 300, num_classes+1]
            # rois:         建议框的坐标      [B, 300, 4]
            # roi_indices:  建议框的index    [B, 300]
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            #---------------------------------#
            #   利用主干网络提取特征
            #---------------------------------#
            base_feature    = self.extractor(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            #---------------------------------#
            #   获得建议框
            #---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            #---------------------------------------#
            #   获得classifier的分类结果和回归结果
            #---------------------------------------#
            roi_cls_locs, roi_scores    = self.head(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
