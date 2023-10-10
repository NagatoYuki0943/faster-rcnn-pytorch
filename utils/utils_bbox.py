import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms

#-----------------------------------#
#   将RPN网络预测结果转化成尚未筛选的建议框
#-----------------------------------#
def loc2bbox(src_bbox, loc):
    """
    src_bbox: 先验框:xyxy
    loc:      先验框调整参数
    return:   建议框:xyxy
    """
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    #-----------------------------------#
    #   计算先验框宽高,中心
    #-----------------------------------#
    src_width   = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)      # x2-x1
    src_height  = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)      # y2-y1
    src_ctr_x   = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width     # x1+0.5w
    src_ctr_y   = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height    # y1+0.5h

    #-----------------------------------#
    #   获取调整参数
    #   每隔4个取一次
    #-----------------------------------#
    dx          = loc[:, 0::4]
    dy          = loc[:, 1::4]
    dw          = loc[:, 2::4]
    dh          = loc[:, 3::4]

    #-----------------------------------#
    #   调整先验框中心和宽高
    #-----------------------------------#
    ctr_x = dx * src_width  + src_ctr_x     # 中心调整就是 预测值*先验框宽高+先验框中心坐标
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width           # 宽高调整就是 预测值的指数*先验框宽高
    h = torch.exp(dh) * src_height

    #-----------------------------------#
    #   将调整后的中心和宽高转换为左上角,右下角坐标
    #-----------------------------------#
    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox

"""建议框解码"""
class DecodeBox():
    def __init__(self, std, num_classes):
        self.std            = std
        self.num_classes    = num_classes + 1

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = 0.3, confidence = 0.5):
        """
        roi_cls_locs: 建议框的调整参数      [B, 300, num_classes*4]  针对每个类别都预测一个框,和yolo只预测1个框和这个框的种类不同
        roi_scores:   建议框的种类得分      [B, 300, num_classes]
        rois:         建议框的坐标          [B, 300, 4]
        image_shape:  原图大小              [1330, 1330]
        input_shape:  调整短边为600后的大小   [600, 600]
        """
        results = []
        bs      = len(roi_cls_locs)
        #--------------------------------#
        #   [batch_size, num_rois, 4] [B, 300, 4]
        #--------------------------------#
        rois    = rois.view((bs, -1, 4))
        #----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(bs):
            #----------------------------------------------------------#
            #   对回归参数进行reshape,改变回归参数数量级
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_locs[i] * self.std                        # [300, num_classes*4]
            #----------------------------------------------------------#
            #   第一维度是建议框的数量，第二维度是每个种类
            #   第三维度是对应种类的调整参数
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])       # [300, num_classes*4] -> [300, num_classes, 4]

            #-------------------------------------------------------------#
            #   利用classifier网络的预测结果对建议框进行调整获得预测框
            #   num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4 重复num_classes次预测框,之后使用建议框的调整参数分别调整
            #-------------------------------------------------------------#
            roi         = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)   # [300, 4] -> [300, 1, 4] -> [300, num_classes, 4]
            #-------------------------------------------------------------#
            #   loc2bbox: 对建议框进行调整获得预测框
            #-------------------------------------------------------------#
            cls_bbox    = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))  # [300, num_classes, 4] & [300, num_classes, 4] = [300*num_classes, 4]
            cls_bbox    = cls_bbox.view([-1, (self.num_classes), 4])        # [300*num_classes, 4] -> [300, num_classes, 4]
            #-------------------------------------------------------------#
            #   对预测框进行归一化，调整到0-1之间
            #-------------------------------------------------------------#
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            #-------------------------------------------------------------#
            #   取出建议框种类得分,并取softmax
            #-------------------------------------------------------------#
            roi_score   = roi_scores[i]                 # [300, num_classes]
            prob        = F.softmax(roi_score, dim=-1)

            results.append([])
            # 从1开始,因为0代表了背景
            for c in range(1, self.num_classes):
                #--------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                #--------------------------------#
                c_confs     = prob[:, c]                # 根据下标取类型 [300]
                c_confs_m   = c_confs > confidence      # 返回 矩阵中为True/False

                if len(c_confs[c_confs_m]) > 0:
                    #-----------------------------------------#
                    #   取出得分高于confidence的框
                    #-----------------------------------------#
                    boxes_to_process = cls_bbox[c_confs_m, c]   # [keep, 4]
                    confs_to_process = c_confs[c_confs_m]       # [keep]

                    #-----------------------------------------#
                    #   非极大抑制
                    #-----------------------------------------#
                    keep = nms(
                        boxes_to_process,   # 框的坐标x1,y1,x2,y2
                        confs_to_process,   # 置信度
                        nms_iou
                    )
                    #-----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    #-----------------------------------------#
                    good_boxes  = boxes_to_process[keep]            # [good, 4]
                    confs       = confs_to_process[keep][:, None]   # [good, 1]
                    labels      = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1)) # [good, 1]
                    #-----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。 [good, 6]
                    #-----------------------------------------#
                    c_pred      = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)

            #-----------------------------------------#
            # 调整为相对于原图的形式
            #-----------------------------------------#
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        #-----------------------------------------------#
        #   results = [ 每张图片
        #               [ 每种类别
        #                  [x1, y1, x2, y2, confs(置信度), labels(种类预测值)],
        #                   ...
        #               ]
        #             ]
        #-----------------------------------------------#
        return results

