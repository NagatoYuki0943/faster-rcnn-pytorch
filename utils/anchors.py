"""
框的示例
"""

import numpy as np

#--------------------------------------------#
#   生成基础的先验框
#   三个竖着的长方形,三个正方形,三个横着的正方形
#--------------------------------------------#
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base

#--------------------------------------------#
#   对基础先验框进行拓展对应到所有特征点上
#--------------------------------------------#
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """_summary_

    Args:
        anchor_base (np.ndarray): 9个先验框的初始形状 [9, 4]
        feat_stride (int):        步长              16
        height (int):             共享特征层的高     38
        width (int):              共享特征层的宽     38

    Returns:
        np.ndarray: [12996, 4]
    """
    #---------------------------------#
    #   计算网格中心点,相对于
    #---------------------------------#
    shift_x             = np.arange(0, width * feat_stride, feat_stride)    # [38]
    shift_y             = np.arange(0, height * feat_stride, feat_stride)   # [38]
    shift_x, shift_y    = np.meshgrid(shift_x, shift_y) # [38, 38], [38, 38] 38 * 38 = 1444
    # 堆叠xyxy代表调整值的顺序为xywh
    shift               = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)   # [1444, 4] ravel == flatten

    #---------------------------------#
    #   每个网格点上的9个先验框
    #---------------------------------#
    A       = anchor_base.shape[0]  # 9
    K       = shift.shape[0]        # 1444
    anchor  = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4)) # ([9, 4] -> [1, 9, 4]) + ([1444, 4] -> [1444, 1, 4]) = [1444, 9, 4]
    #---------------------------------#
    #   所有的先验框
    #---------------------------------#
    anchor  = anchor.reshape((K * A, 4)).astype(np.float32) # [1444, 9, 4] -> [1444*9, 4]  1444*9 = 12996
    return anchor

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nine_anchors = generate_anchor_base()
    print(nine_anchors)

    height, width, feat_stride  = 38,38,16
    anchors_all                 = _enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)
    print(np.shape(anchors_all))

    fig     = plt.figure(figsize=(9, 9))
    ax      = fig.add_subplot(111)  # 1行1列第1个
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    shift_x = np.arange(0, width * feat_stride,  feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x,shift_y)
    box_widths  = anchors_all[:,2]-anchors_all[:,0]
    box_heights = anchors_all[:,3]-anchors_all[:,1]

    for i in [108, 109, 110, 111, 112, 113, 114, 115, 116]:
        rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)
    plt.show()
