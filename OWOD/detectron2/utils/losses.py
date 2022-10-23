import torch
import math
from detectron2.layers import nonzero_tuple

##----------- Added by Johan on 1/3/2020 ------------------------------------------------------
##----------- Start of code -------------------------------------------------------------------

def compute_diou_mmdet(pred, target, eps=1e-7):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.
    Code is modified from https://github.com/Zzh-tju/DIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw ** 2 + ch ** 2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious

    loss = loss.sum()

    return loss


def compute_diou_fvcore(pred, gt_boxes):

    boxes1 = pred
    boxes2 = gt_boxes
    eps = 1e-7

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    # area_c = (xc2 - xc1) * (yc2 - yc1)
    # miouk = iouk - ((area_c - unionk) / (area_c + eps))

    # set_trace()
    c = 0
    d = 0
    # if self.cfg.MODEL.ROI_BOX_HEAD.LOSS_BOX_WEIGHT == 10:
    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    # else:
    # c = torch.sqrt(((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps)
    # d = torch.sqrt(((x_p - x_g) ** 2) + ((y_p - y_g) ** 2))

    u = d / c
    diouk = iouk - u

    loss = (1 - diouk)

    #Default reduction: sum
    loss = loss.sum()

    return loss

def bbox_transform(deltas, weights, scale_clamp):
    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    dw = torch.clamp(dw, max=scale_clamp)
    dh = torch.clamp(dh, max=scale_clamp)

    pred_ctr_x = dx
    pred_ctr_y = dy
    pred_w = torch.exp(dw)
    pred_h = torch.exp(dh)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    return x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)

def compute_diou(output_delta, target_delta, weights , scale_clamp):

    #Note: This version of DIOU uses delta values instead of actual bboxes
    #I found delta values more efficient for our case
    # output_delta = self.pred_proposal_deltas
    # target_delta = self.box2box_transform.get_deltas(
    #                   self.proposals.tensor, self.gt_boxes.tensor
    # )

    # # Handled at function call
    # #Borrowed from sl1. Earlier verison used mask code
    # #This section simply set's mask = True for those coordinates bounding boxes
    # #which have an IOU above threshold (as per current Faster thr is 50, For
    # #Cascade it is 50, 60. 70) with gt_boxes
    # bg_class_ind = self.pred_class_logits.shape[1] - 1
    # box_dim = target_delta.size(1)  # 4 or 5
    # fg_inds = torch.nonzero(
    #     (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind), as_tuple=True
    # )[0]
    # gt_class_cols = torch.arange(box_dim, device=self.pred_proposal_deltas.device)
    # output_delta = output_delta[fg_inds[:, None], gt_class_cols]
    # target_delta = target_delta[fg_inds]

    #Note: We use delta values here as per the orignal authors code
    #Delta values are : (center_x, center_y, w, h).
    #Here the bbox_transform unlike Detectron2's Box2BoxTransform get_deltas converts
    #the delta coordinates to x1, y1, x2, y2. These coordinates are still deltas but
    #they are used in calculating DIOU loss and hence the bbox_transform function is still used
    x1, y1, x2, y2 = bbox_transform(output_delta, weights, scale_clamp)
    x1g, y1g, x2g, y2g = bbox_transform(target_delta, weights, scale_clamp)


    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    #Instersction of two boxes
    intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)   #Optimized

    #Union of two boxes
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    #Note both of the below distances do not use square root.
    #As per the authors advice the gradient would have to calculate square
    #root as well. Hence it hasn't been used here.
    #Length of largest diagonal of the polygon covering both boxes.
    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
    #Distance between center points.
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    diouk = iouk - u

    # diouk = (1 - diouk).sum() / self.gt_classes.numel()
    # diouk = diouk * self.cfg.MODEL.ROI_BOX_HEAD.LOSS_BOX_WEIGHT
    loss = (1 - diouk)

    #Default reduction: sum
    loss = loss.sum()

    return loss


def compute_ciou_mmdet(pred, target, eps=1e-7):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw ** 2 + ch ** 2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right

    factor = 4 / math.pi ** 2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v ** 2 / (1 - ious + v))
    loss = 1 - cious

    loss = loss.sum()

    return loss

# def compute_ciou(self):
#
#     output = self.pred_proposal_deltas
#     target = self.box2box_transform.get_deltas(
#         self.proposals.tensor, self.gt_boxes.tensor
#     )
#
#     x1, y1, x2, y2 = self.bbox_transform(output, self.box2box_transform.weights)
#     x1g, y1g, x2g, y2g = self.bbox_transform(target, self.box2box_transform.weights)
#
#     # box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
#     # device = self.pred_proposal_deltas.device
#     # bg_class_ind = self.pred_class_logits.shape[1] - 1
#     # gt_class_cols = torch.arange(box_dim, device=device)
#     #
#     # fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
#     #
#     # boxes1 = self._predict_boxes()[fg_inds[:, None], gt_class_cols]
#     # boxes2 = self.gt_boxes.tensor[fg_inds]
#
#     # x1, y1, x2, y2 = boxes1.unbind(dim=-1)
#     # x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
#
#     # set_trace()
#
#     x2 = torch.max(x1, x2)
#     y2 = torch.max(y1, y2)
#     # assert (x2 >= x1).all(), "bad box: x1 larger than x2"
#     # assert (y2 >= y1).all(), "bad box: y1 larger than y2"
#     w_pred = x2 - x1
#     h_pred = y2 - y1
#     w_gt = x2g - x1g
#     h_gt = y2g - y1g
#
#     x_center = (x2 + x1) / 2
#     y_center = (y2 + y1) / 2
#     x_center_g = (x1g + x2g) / 2
#     y_center_g = (y1g + y2g) / 2
#
#     xkis1 = torch.max(x1, x1g)
#     ykis1 = torch.max(y1, y1g)
#     xkis2 = torch.min(x2, x2g)
#     ykis2 = torch.min(y2, y2g)
#
#     xc1 = torch.min(x1, x1g)
#     yc1 = torch.min(y1, y1g)
#     xc2 = torch.max(x2, x2g)
#     yc2 = torch.max(y2, y2g)
#
#     # intsctk = torch.zeros(x1.size()).to(output)
#     intsctk = torch.zeros_like(x1)
#     mask = (ykis2 > ykis1) * (xkis2 > xkis1)
#     intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
#     unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
#     iouk = intsctk / unionk
#
#     c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
#     d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
#     u = d / c
#
#     v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
#     with torch.no_grad():
#         S = 1 - iouk
#         alpha = v / (S + v)
#     ciouk = iouk - (u + alpha * v)
#
#     bg_class_ind = self.pred_class_logits.shape[1] - 1
#
#     fg_inds = torch.nonzero(
#         (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind), as_tuple=True
#     )[0]
#
#     ciouk = (1 - ciouk[fg_inds]).sum() / self.gt_classes.numel()
#
#     ciouk = ciouk * self.cfg.MODEL.ROI_BOX_HEAD.LOSS_BOX_WEIGHT
#
#     return ciouk