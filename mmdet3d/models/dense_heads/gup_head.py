# Copyright (c) OpenMMLab. All rights reserved.
import pdb
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import abstractmethod

import numpy as np
from mmdet.core import multi_apply
from mmdet.core.bbox.builder import build_bbox_coder
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum,
                                                get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from .anchor_free_mono3d_head import AnchorFreeMono3DHead
from mmcv.cnn import bias_init_with_prob, normal_init, constant_init
from mmcv.runner import force_fp32
import torch.distributions as distributions
import torchvision.ops.roi_align as roi_align
from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr, points_cam2img
import neptune
import os


def cube_root(x):
    minus_index = x<0
    return torch.abs(x).pow(1/3)*(minus_index*(-1))

@HEADS.register_module()
class GUPHead(AnchorFreeMono3DHead):
    """Anchor-free head used in `SMOKE <https://arxiv.org/abs/2002.10111>`_

    .. code-block:: none

                /-----> 3*3 conv -----> 1*1 conv -----> cls
        feature
                \-----> 3*3 conv -----> 1*1 conv -----> reg

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        dim_channel (list[int]): indices of dimension offset preds in
            regression heatmap channels.
        ori_channel (list[int]): indices of orientation offset pred in
            regression heatmap channels.
        bbox_coder (:obj:`CameraInstance3DBoxes`): Bbox coder
            for encoding and decoding boxes.
        loss_cls (dict, optional): Config of classification loss.
            Default: loss_cls=dict(type='GaussionFocalLoss', loss_weight=1.0).
        loss_bbox (dict, optional): Config of localization loss.
            Default: loss_bbox=dict(type='L1Loss', loss_weight=10.0).
        loss_dir (dict, optional): Config of direction classification loss.
            In SMOKE, Default: None.
        loss_attr (dict, optional): Config of attribute classification loss.
            In SMOKE, Default: None.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict): Initialization config dict. Default: None.
    """  # noqa: E501

    def __init__(self,
                 bbox_coder,
                 mean_size,
                 id2cat,
                 max_objs=50,
                 downsample=4,
                 loss_offset2d = dict(type='L1Loss', loss_weight=1.0),
                 loss_offset3d = dict(type='L1Loss', loss_weight=1.0),
                 loss_size2d = dict(type='L1Loss', loss_weight=1.0),
                 loss_size3d = dict(type='UncertainL1Loss', loss_weight=1.4142, alpha=1/1.4142, beta=0.5),
                 loss_depth = dict(type='UncertainL1Loss', loss_weight=1.4142, alpha=1/1.4142, beta=0.5),
                 loss_angle_bin = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_angle_res = dict(type='L1Loss', loss_weight=1.0),
                 clsmap_type = 'hard',
                 basedim3d_type = 'hard',
                 uncertainty_type = 'GeU',
                 use_3d_center=True,
                 use_3d_nms = False,
                 use_iounc = False,
                 heuristic_loss_scheme = None,
                 **kwargs):
        super().__init__(
            **kwargs)
        self.nms2d_kenerl = 3
        self.downsample = downsample
        self.max_objs = max_objs
        self.mean_size = torch.tensor([mean_size[cat] for cat in id2cat])
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_offset2d = build_loss(loss_offset2d)
        self.loss_offset3d = build_loss(loss_offset3d)
        self.loss_size2d = build_loss(loss_size2d)
        self.loss_size3d = build_loss(loss_size3d)
        self.loss_depth = build_loss(loss_depth)
        self.loss_angle_bin = build_loss(loss_angle_bin)
        self.loss_angle_res = build_loss(loss_angle_res)
        if heuristic_loss_scheme is not None:
            self.heuristic_loss_scheme = build_loss(heuristic_loss_scheme)
        self.use_3d_center = use_3d_center
        self.use_3d_nms = use_3d_nms
        self.use_iounc = use_iounc
        # type of class map 
        assert(clsmap_type=='hard' or clsmap_type=='soft')
        assert(basedim3d_type=='hard' or basedim3d_type=='soft')
        assert(uncertainty_type=='GeU' or uncertainty_type=='GeU++')
        self.clsmap_type = clsmap_type
        self.basedim3d_type = basedim3d_type
        self.uncertainty_type = uncertainty_type

    def forward(self, feats, img_metas, targets=None):
        ori_featmaps = feats[0]
        DEVICE, (BATCH_SIZE, _, HEIGHT, WIDE) = ori_featmaps.device, ori_featmaps.shape
        featmaps_for_2dheads = torch.cat([ori_featmaps, ori_featmaps.new_zeros((BATCH_SIZE, 2+self.num_classes, HEIGHT, WIDE))],1)
        # meta info
        trans_mats = torch.stack([
            ori_featmaps.new_tensor(img_meta['trans_mat'])
            for img_meta in img_metas
        ])
        calibs = torch.stack([
            ori_featmaps.new_tensor(img_meta['cam2img'])
            for img_meta in img_metas
        ])

        # cls score
        cls_map = self.conv_cls(self.conv_cls_prev[0](featmaps_for_2dheads))
        cls_score = cls_map.sigmoid()
        cls_score = torch.clamp(cls_score, min=1e-4, max=1 - 1e-4)
        attr_pred = None
        if self.pred_attrs:
            attr_pred = self.conv_attr(self.conv_attr_prev[0](featmaps_for_2dheads))       
        # 2d regression
        offset_2d = self.conv_regs[0](self.conv_reg_prevs[0][0](featmaps_for_2dheads))
        size_2d = self.conv_regs[1](self.conv_reg_prevs[1][0](featmaps_for_2dheads))

        if self.uncertainty_type == 'GeU++':
            size_2d, size_2d_un = size_2d[:,:2], size_2d[:,2:]

        bbox2d_maps_featmaps, bbox2d_maps_ori = self.decode_bbox2d_maps(offset_2d, size_2d, trans_mats)
        bbox2d_maps_featmaps = torch.cat([torch.arange(BATCH_SIZE).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat([1,1,HEIGHT,WIDE]).type(torch.float).to(DEVICE),bbox2d_maps_featmaps],1)

        if targets is not None:   #extract train structure in the train (only) and the val mode
            inds,cls_ids = targets['indices'],targets['cls_ids']
            masks = targets['mask']
        else:    #extract test structure in the test (only) and the val mode
            if self.use_3d_nms:
                _, inds, cls_ids, _, _ = get_topk_from_heatmap(cls_score, k= 2*self.max_objs)
            else:
                center_heatmap_pred = get_local_maximum(cls_score, kernel=self.nms2d_kenerl)
                _, inds, cls_ids, _, _ = get_topk_from_heatmap(center_heatmap_pred, k=self.max_objs)
            masks = inds.new_ones(inds.size()).type(torch.uint8)
        # get roi features
        num_masked_bin = masks.sum()
        if num_masked_bin>0:
            # roi representation
            bbox2ds_fm = transpose_and_gather_feat(bbox2d_maps_featmaps, inds)[masks]
            roi_features = roi_align(ori_featmaps, bbox2ds_fm,[7,7])
            with torch.no_grad():
                roi_calibs = calibs[bbox2ds_fm[:,0].long()]
                roi_trans_mats = trans_mats[bbox2ds_fm[:,0].long()]
                bbox2d_maps_ori = transpose_and_gather_feat(bbox2d_maps_ori, inds)[masks]
                # catelog maps
                if self.clsmap_type == 'hard':
                    cls_hots = torch.zeros(num_masked_bin,self.num_classes).to(DEVICE)
                    cls_hots[torch.arange(num_masked_bin).to(DEVICE),cls_ids[masks].long()] = 1.0
                elif self.clsmap_type == 'soft':
                    cls_hots = transpose_and_gather_feat(cls_score, inds)[masks]
                # coordinate maps
                if roi_calibs.shape[-1]==4: 
                    coords_in_camera_coord = torch.cat([(roi_calibs.inverse() @ torch.cat([bbox2d_maps_ori[:,:2],bbox2d_maps_ori.new_ones([num_masked_bin,2])],-1).unsqueeze(-1))[:,:2, 0],
                                                    (roi_calibs.inverse() @ torch.cat([bbox2d_maps_ori[:,2:],bbox2d_maps_ori.new_ones([num_masked_bin,2])],-1).unsqueeze(-1))[:,:2, 0]],-1)
                else:
                    coords_in_camera_coord = torch.cat([(roi_calibs.inverse() @ torch.cat([bbox2d_maps_ori[:,:2],bbox2d_maps_ori.new_ones([num_masked_bin,1])],-1).unsqueeze(-1))[:,:2, 0],
                                                    (roi_calibs.inverse() @ torch.cat([bbox2d_maps_ori[:,2:],bbox2d_maps_ori.new_ones([num_masked_bin,1])],-1).unsqueeze(-1))[:,:2, 0]],-1)
                coord_maps = torch.cat([torch.cat([coords_in_camera_coord[:,0:1]+i*(coords_in_camera_coord[:,2:3]-coords_in_camera_coord[:,0:1])/6 for i in range(7)],-1).unsqueeze(1).repeat([1,7,1]).unsqueeze(1),
                                        torch.cat([coords_in_camera_coord[:,1:2]+i*(coords_in_camera_coord[:,3:4]-coords_in_camera_coord[:,1:2])/6 for i in range(7)],-1).unsqueeze(2).repeat([1,1,7]).unsqueeze(1)],1)
            # cat roi feats
            roi_features = torch.cat([roi_features,coord_maps,cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1,1,7,7])],1)

            # roi head regressions
            depth = self.conv_regs[2](F.adaptive_avg_pool2d(self.conv_reg_prevs[2][0](roi_features),1)).squeeze(-1).squeeze(-1)
            offset_3d = self.conv_regs[3](F.adaptive_avg_pool2d(self.conv_reg_prevs[3][0](roi_features),1)).squeeze(-1).squeeze(-1)
            size_3d = self.conv_regs[4](F.adaptive_avg_pool2d(self.conv_reg_prevs[4][0](roi_features),1)).squeeze(-1).squeeze(-1)
            heading = self.conv_regs[5](F.adaptive_avg_pool2d(self.conv_reg_prevs[5][0](roi_features),1)).squeeze(-1).squeeze(-1)
            # gup head
            size_3d_offset, size_3d_offset_un = size_3d[:,:3], size_3d[:,3:]
            depth_bias, depth_bias_log_var = depth[:,:1], depth[:,1:]
            focal_lengths = roi_calibs[:,0,0].unsqueeze(-1)

            #h2d = torch.clamp((bbox2ds_fm[:,4:5]-bbox2ds_fm[:,2:3])/roi_trans_mats[:,1,1].unsqueeze(-1),min=1.0)
            h2d = torch.clamp(bbox2d_maps_ori[:,3:4]-bbox2d_maps_ori[:,1:2],min=1.0)
            if self.uncertainty_type == 'GeU++':
                h2d_log_var = transpose_and_gather_feat(size_2d_un[:, 1:2], inds)[masks]
                h2d_log_var = h2d_log_var - 2*roi_trans_mats[:,1,1].unsqueeze(-1).log()

            if self.basedim3d_type == 'hard':
                size3d_base = self.mean_size.to(DEVICE)[cls_ids[masks].long()]
                size3d_log_var = size_3d_offset_un
            elif self.basedim3d_type == 'soft':
                norm_class_dis = transpose_and_gather_feat(cls_map, inds)[masks].softmax(1)
                size3d_base = norm_class_dis @ self.mean_size.to(DEVICE)
                size3d_base_logvar = ((norm_class_dis.unsqueeze(-1) * (self.mean_size.to(DEVICE).unsqueeze(0) - size3d_base.unsqueeze(1))**2).sum(1)+1e-12).log()
                size3d_log_var = torch.logsumexp(torch.cat([size3d_base_logvar.unsqueeze(-1),size_3d_offset_un.unsqueeze(-1)],-1),-1)

            size3d = torch.clamp(size_3d_offset + size3d_base, min=0.01)
            h3d, h3d_log_var = size3d[:,1:2], size3d_log_var[:,1:2]

            depth_geo = h3d*focal_lengths/h2d
            if self.uncertainty_type == 'GeU':
                depth_geo_log_var = h3d_log_var+2*(torch.log(focal_lengths)-torch.log(h2d))
            elif self.uncertainty_type == 'GeU++':
                depth_geo_log_var = torch.log(depth_geo**2) + torch.logsumexp(torch.cat([h3d_log_var-torch.log(h3d**2),h2d_log_var-torch.log(h2d**2)],-1), -1, keepdim=True)

            depth = depth_geo + depth_bias
            depth_log_var = torch.logsumexp(torch.cat([depth_bias_log_var,depth_geo_log_var],-1),-1,keepdim=True)

            depth = torch.cat([depth, 0.5 * depth_log_var],dim=1)
            size_3d = torch.cat([size3d, 0.5 * size3d_log_var],dim=1)
            if self.uncertainty_type == 'GeU++':
                size_2d = torch.cat([size_2d, 0.5 * size_2d_un],dim=1)
        else:
            depth = None
            offset_3d = None
            size_3d = None
            heading = None
        pred_reg = [offset_2d, size_2d, depth, offset_3d, size_3d, heading]

        return cls_score, pred_reg, attr_pred

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_3d (list[Tensor]): 3D ground truth bboxes of the image,
                shape (num_gts, self.bbox_code_size).
            gt_labels_3d (list[Tensor]): 3D ground truth labels of each box,
                shape (num_gts,).
            centers2d (list[Tensor]): Projected 3D center of each box,
                shape (num_gts, 2).
            depths (list[Tensor]): Depth of projected 3D center of each box,
                shape (num_gts,).
            attr_labels (list[Tensor]): Attribute labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """      
        if gt_labels is None:
            loss_inputs = (x, gt_bboxes, gt_bboxes_3d, centers2d, depths,
                                  attr_labels, img_metas)
        else:
            loss_inputs = (x, gt_bboxes, gt_labels, gt_bboxes_3d,
                                  gt_labels_3d, centers2d, depths, attr_labels,
                                  img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def get_bboxes(self, feat_maps, img_metas, rescale=None):
        """Generate bboxes from bbox head predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
            bbox_preds (list[Tensor]): Box regression for each scale.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[tuple[:obj:`CameraInstance3DBoxes`, Tensor, Tensor, None]]:
                Each item in result_list is 4-tuple.
        """
        cam2imgs = torch.stack([
            feat_maps[0].new_tensor(img_meta['cam2img'])
            for img_meta in img_metas
        ])
        trans_mats = torch.stack([
            feat_maps[0].new_tensor(img_meta['trans_mat'])
            for img_meta in img_metas
        ])

        cls_scores, bbox_preds, attr_preds = self(feat_maps, img_metas)

        batch_bboxes_2d, \
        batch_bboxes_3d, \
        batch_scores, \
        batch_scores_3d, \
        batch_topk_labels, \
        batch_attrs, \
        bbox_preds, \
        all_cls_scores = \
            self.decode_heatmap(
            cls_scores,  # 0 is level index
            bbox_preds,  # 0 is level index
            attr_preds,
            img_metas,
            cam2imgs=cam2imgs,
            trans_mats=trans_mats,
            kernel=self.nms2d_kenerl)
        

        result_list = []
        for img_id in range(len(img_metas)):

            bboxes_2d = batch_bboxes_2d[img_id]
            bboxes_3d = batch_bboxes_3d[img_id]
            scores = batch_scores[img_id]
            scores_3d = batch_scores_3d[img_id]
            labels = batch_topk_labels[img_id]
            all_cls_score = all_cls_scores[img_id]
            attrs = batch_attrs[img_id]
                
 
            if bboxes_3d.shape[-1]!=self.bbox_code_size: # add velo
                if self.pred_velo: 
                    velo = bbox_preds[-1][img_id][keep_idx]
                else:
                    velo = bboxes_3d.new_zeros([bboxes_3d.shape[0],self.bbox_code_size-bboxes_3d.shape[1]])
                bboxes_3d = torch.cat([bboxes_3d,velo],1)

            # 2d box nms
            if self.use_3d_nms:
                bboxes_for_nms = xywhr2xyxyr(img_metas[img_id]['box_type_3d'](
                    bboxes_3d,
                    box_dim=self.bbox_code_size,
                    origin=(0.5, 0.5, 0.5)).bev)
                all_cls_score = torch.cat([all_cls_score,all_cls_score.new_zeros((all_cls_score.shape[0],1))],1)
                results = box3d_multiclass_nms(bboxes_3d, bboxes_for_nms, all_cls_score, 0.01, self.max_objs, self.test_cfg, mlvl_bboxes2d=bboxes_2d, mlvl_attr_scores=attrs)
                bboxes_3d, scores, labels, attrs, bboxes_2d = results
            else:
                keep_idx_score = scores > 0.2
                keep_dim_3d = (1-(bboxes_3d[:,3:6]<=0.0).sum(-1)).bool()
                keep_idx = keep_idx_score * keep_dim_3d
                bboxes_2d = bboxes_2d[keep_idx]
                bboxes_3d = bboxes_3d[keep_idx]
                scores = scores[keep_idx]
                labels = labels[keep_idx]
                attrs = attrs[keep_idx]
            if attrs.shape[0]>0:attrs = attrs.argmax(1)
            bboxes_3d = img_metas[img_id]['box_type_3d'](
                bboxes_3d, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
            if self.pred_bbox2d: 
                result_list.append((bboxes_3d, scores, labels, attrs, bboxes_2d))
            else:
                result_list.append((bboxes_3d, scores, labels, attrs))
        return result_list

    def decode_heatmap(self,
                       cls_score,
                       reg_pred,
                       attr_pred,
                       img_metas,
                       cam2imgs,
                       trans_mats,
                       kernel=3,
                       **kwargs):
        """Transform outputs into detections raw bbox predictions.

        Args:
            class_score (Tensor): Center predict heatmap,
                shape (B, num_classes, H, W).
            reg_pred (Tensor): Box regression map.
                shape (B, channel, H , W).
            img_metas (List[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cam2imgs (Tensor): Camera intrinsic matrixs.
                shape (B, 4, 4)
            trans_mats (Tensor): Transformation matrix from original image
                to feature map.
                shape: (batch, 3, 3)
            topk (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of SMOKEHead, containing
               the following Tensors:
              - batch_bboxes (Tensor): Coords of each 3D box.
                    shape (B, k, 7)
              - batch_scores (Tensor): Scores of each 3D box.
                    shape (B, k)
              - batch_topk_labels (Tensor): Categories of each 3D box.
                    shape (B, k)
        """
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        device, (bs, cls_num, feat_h, feat_w) = cls_score.device, cls_score.shape

        # nms
        if self.use_3d_nms:
            batch_scores_2d, batch_index, batch_topk_labels, topk_ys, topk_xs = get_topk_from_heatmap(cls_score, k=2*self.max_objs)
            regression = [transpose_and_gather_feat(_, batch_index) if i<2 else _.reshape(bs, 2*self.max_objs, -1) for i, _ in enumerate(reg_pred)]
        else:
            center_heatmap_pred = get_local_maximum(cls_score, kernel=kernel)
            batch_scores_2d, batch_index, batch_topk_labels, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=self.max_objs)
            regression = [transpose_and_gather_feat(_, batch_index) if i<2 else _.reshape(bs, self.max_objs, -1) for i, _ in enumerate(reg_pred)]
        
        if self.num_classes==7 and self.bbox_code_size==7:  #kitti all cls
            keep_index = (batch_topk_labels<=2)[0]
            batch_scores_2d = batch_scores_2d[:,keep_index]
            batch_index = batch_index[:,keep_index]
            batch_topk_labels = batch_topk_labels[:,keep_index]
            topk_ys = topk_ys[:,keep_index]
            topk_xs = topk_xs[:,keep_index]
            regression = [_[:,keep_index] for _ in regression]

        #if self.pred_attrs:
        #    attr_pred = transpose_and_gather_feat(attr_pred, batch_index)
        #else:
        #    attr_pred = batch_scores_2d.new_zeros([batch_index.shape[0],batch_index.shape[1],self.num_attrs])
        attr_pred = batch_scores_2d.new_zeros([batch_index.shape[0],batch_index.shape[1],self.num_attrs])

        # offset_2d, size_2d, depth, offset_3d, size_3d, heading
        points = torch.stack([topk_xs, topk_ys.float()], dim=-1)

        # decode heads
        offset_2d, size_2d, depth, offset_3d, size_3d, heading = regression
        size_3d, size_3d_log_std = size_3d[:,:,:3], size_3d[:,:,3:]
        depth, depth_log_std = depth[:,:,:1], depth[:,:,1:]
        size_2d, size_2d_log_std = size_2d[:,:,:2], size_2d[:,:,2:]

        regression = [offset_2d, size_2d, depth, offset_3d, size_3d, heading]
        regression_log_stds = [None, size_2d_log_std, depth_log_std, None, size_3d_log_std, None]
        
        batch_bboxes_2d, batch_bboxes_3d = self.bbox_coder.decode(
            regression, points, batch_topk_labels, cam2imgs, trans_mats)

        if self.use_iounc:
            '''
            if cls_num==3: #KITTI
                IoU = torch.tensor([0.5,0.5,0.7]).to(device)[batch_topk_labels]/10
            else: #nuscenes
                IoU = 0.7
            cos, sin = batch_bboxes_3d[:,:,-1].cos(),batch_bboxes_3d[:,:,-1].sin()
            rot_inv = torch.stack([torch.stack([cos,cos.new_zeros(cos.shape),sin],2),torch.stack([cos.new_zeros(cos.shape),cos.new_ones(cos.shape),cos.new_zeros(cos.shape)],2),torch.stack([-sin,cos.new_zeros(cos.shape),cos],2)],2).inverse()
            center_dir = F.normalize(rot_inv @ batch_bboxes_3d[:,:,:3].unsqueeze(-1),2,2)[:,:,:,0].abs()
            bbox3d_dim = batch_bboxes_3d[:,:,3:6]
            #equation solver 
            a = (1+IoU)*(-center_dir[:,:,0]*center_dir[:,:,1]*center_dir[:,:,2])
            b = (1+IoU)*(bbox3d_dim[:,:,0]*center_dir[:,:,1]*center_dir[:,:,2]+\
                          bbox3d_dim[:,:,1]*center_dir[:,:,0]*center_dir[:,:,2]+\
                          bbox3d_dim[:,:,2]*center_dir[:,:,0]*center_dir[:,:,1])
            c = (1+IoU)*(-(bbox3d_dim[:,:,0]*bbox3d_dim[:,:,1]*center_dir[:,:,2]+\
                         bbox3d_dim[:,:,0]*bbox3d_dim[:,:,2]*center_dir[:,:,1]+\
                         bbox3d_dim[:,:,2]*bbox3d_dim[:,:,1]*center_dir[:,:,0]))
            d = (1-IoU)*bbox3d_dim[:,:,0]*bbox3d_dim[:,:,1]*bbox3d_dim[:,:,2]
            cubic_solver_alpha = (b*c)/(6*a**2) - (b**3)/(27*a**3) - d/(2*a)
            cubic_solver_beta = c/(3*a)-(b**2)/(9*a**2)
            cubic_solver_delta = (cubic_solver_alpha)**2+(cubic_solver_beta)**3
            dir_ratio_cond1 = -b/(3*a)+\
                              cube_root(cubic_solver_alpha+torch.sqrt(cubic_solver_delta))+\
                              cube_root(cubic_solver_alpha-torch.sqrt(cubic_solver_delta))
            dir_ratio_cond2 = -b/(3*a)+2*torch.sqrt(-cubic_solver_beta)*torch.cos((torch.arccos(cubic_solver_alpha/(-cubic_solver_beta)**(3/2))+2*torch.pi)/3)
            dir_ratio_cond1[cubic_solver_delta<0] = 0.0; dir_ratio_cond2[cubic_solver_delta>0] = 0.0
            dir_ratio = dir_ratio_cond1 + dir_ratio_cond2
            depth_range = (center_dir*dir_ratio.unsqueeze(-1))[:,:,2]            
            #Aa = bbox3d_dim - dir_ratio.unsqueeze(-1)*center_dir
            #evaluation = Aa[:,:,0]*Aa[:,:,1]*Aa[:,:,2]/(2*bbox3d_dim[:,:,0]*bbox3d_dim[:,:,1]*bbox3d_dim[:,:,2] - Aa[:,:,0]*Aa[:,:,1]*Aa[:,:,2])
            #((b*c)/(6*a**2) - (b**3)/(27*a**3) - d/(2*a))**2+(c/(3*a)-(b**2)/(9*a**2))**3
            '''
            if cls_num==3: #KITTI
                IoU = torch.tensor([0.5,0.5,0.7]).to(device)[batch_topk_labels]
            else: #nuscenes
                IoU = 0.7
            cos, sin = batch_bboxes_3d[:,:,-1].cos(),batch_bboxes_3d[:,:,-1].sin()
            rot_inv = torch.stack([torch.stack([cos,sin],2),torch.stack([-sin,cos],2)],2).inverse()
            center_dir = F.normalize(rot_inv @ batch_bboxes_3d[:,:,[0,2]].unsqueeze(-1),2,2)[:,:,:,0].abs()
            bev_box = batch_bboxes_3d[:,:,[3,5]]
            Aa = (1+IoU)*center_dir[:,:,0]*center_dir[:,:,1]
            Bb = -(1+IoU)*(bev_box[:,:,0]*center_dir[:,:,1]+bev_box[:,:,1]*center_dir[:,:,0])
            Cc = (1-IoU)*bev_box[:,:,0]*bev_box[:,:,1]
            dir_ratio = (-Bb-(Bb**2-4*Aa*Cc).sqrt())/(2*Aa+1e-12)       
            depth_range = F.normalize(batch_bboxes_3d[:,:,[0,2]],2,2)[:,:,1]*dir_ratio
            if 'L1' in self.loss_depth.__class__.__name__:
                batch_scores_3d = 1-torch.exp(-depth_range*np.sqrt(2)/regression_log_stds[2][:,:,0].exp())  #Laplacian PDF
            else:
                batch_scores_3d = torch.erf(depth_range/(regression_log_stds[2][:,:,0].exp()*np.sqrt(2)))  #Gaussian PDF

        else:
            batch_scores_3d = (-regression_log_stds[2][:,:,0].exp()).exp()

        batch_scores = batch_scores_2d * batch_scores_3d

        all_cls_scores = transpose_and_gather_feat(cls_score, batch_index) * batch_scores_3d.unsqueeze(-1)
        if self.num_classes==7 and self.bbox_code_size==7: 
            all_cls_scores = all_cls_scores[:,:,:3]
        
        batch_bboxes_2d = torch.cat([batch_bboxes_2d,
                                     batch_scores_2d.unsqueeze(-1)],-1)
        return batch_bboxes_2d, batch_bboxes_3d, batch_scores, batch_scores_3d, batch_topk_labels, attr_pred, regression, all_cls_scores

    def get_targets(self, gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, gt_attr_labels,
                    centers2d, feat_shape, img_shape, img_metas):
        """Get training targets for batch images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gt,).
            gt_bboxes_3d (list[:obj:`CameraInstance3DBoxes`]): 3D Ground
                truth bboxes of each image,
                shape (num_gt, bbox_code_size).
            gt_labels_3d (list[Tensor]): 3D Ground truth labels of each
                box, shape (num_gt,).
            centers2d (list[Tensor]): Projected 3D centers onto 2D image,
                shape (num_gt, 2).
            feat_shape (tuple[int]): Feature map shape with value,
                shape (B, _, H, W).
            img_shape (tuple[int]): Image shape in [h, w] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple[Tensor, dict]: The Tensor value is the targets of
                center heatmap, the dict has components below:
              - gt_centers2d (Tensor): Coords of each projected 3D box
                    center on image. shape (B * max_objs, 2)
              - gt_labels3d (Tensor): Labels of each 3D box.
                    shape (B, max_objs, )
              - indices (Tensor): Indices of the existence of the 3D box.
                    shape (B * max_objs, )
              - affine_indices (Tensor): Indices of the affine of the 3D box.
                    shape (N, )
              - gt_locs (Tensor): Coords of each 3D box's location.
                    shape (N, 3)
              - gt_dims (Tensor): Dimensions of each 3D box.
                    shape (N, 3)
              - gt_yaws (Tensor): Orientation(yaw) of each 3D box.
                    shape (N, 1)
              - gt_cors (Tensor): Coords of the corners of each 3D box.
                    shape (N, 8, 3)
        """
        # convert the mmdetection style as original gupnet
        # centers2d means the projected 3d center in image coord
        center3ds = [_/self.downsample for _ in centers2d]

        # put gt 3d bboxes to gpu
        gt_bboxes_3d = [
            gt_bbox_3d.tensor.to(centers2d[0].device) for gt_bbox_3d in gt_bboxes_3d
        ]

        # mask_3dbbox3d_proj_corners,
        mask_2d = gt_bboxes[0].new_ones(len(centers2d)).bool()
        if 'affine_aug' not in img_metas[0]:
            mask_3d = gt_bboxes[0].new_zeros(len(centers2d)).bool()
        else:
            mask_3d = torch.stack([
            gt_bboxes[0].new_tensor(
                not img_meta['affine_aug'], dtype=torch.bool)
            for img_meta in img_metas
            ])

        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        # box2ds
        gt_bboxes = [gt_bbox/self.downsample for gt_bbox in gt_bboxes]
        # heatmaps
        heatmap = gt_bboxes[0].new_zeros([bs, self.num_classes, feat_h, feat_w],dtype=torch.float32)
        size_2d = gt_bboxes[0].new_zeros((bs, self.max_objs, 2),dtype=torch.float32)
        size_3d = gt_bboxes[0].new_zeros((bs, self.max_objs, 3),dtype=torch.float32)
        src_size_3d = gt_bboxes[0].new_zeros((bs, self.max_objs, 3),dtype=torch.float32)
        xyz = gt_bboxes[0].new_zeros((bs, self.max_objs, 3),dtype=torch.float32)
        offset_2d = gt_bboxes[0].new_zeros((bs, self.max_objs, 2),dtype=torch.float32)
        offset_3d = gt_bboxes[0].new_zeros((bs, self.max_objs, 2),dtype=torch.float32)
        offset_up_down = gt_bboxes[0].new_zeros((bs, self.max_objs, 4),dtype=torch.float32)
        cls_ids = gt_bboxes[0].new_zeros((bs, self.max_objs),dtype=torch.int64)
        indices = gt_bboxes[0].new_zeros((bs, self.max_objs),dtype=torch.int64)
        mask = gt_bboxes[0].new_zeros((bs, self.max_objs),dtype=torch.bool)
        heading_bin = gt_bboxes[0].new_zeros((bs, self.max_objs, 1),dtype=torch.int64)
        heading_res = gt_bboxes[0].new_zeros((bs, self.max_objs, 1),dtype=torch.float32)
        velo = gt_bboxes[0].new_zeros((bs, self.max_objs, 2),dtype=torch.float32)
        attributes = gt_bboxes[0].new_zeros((bs, self.max_objs),dtype=torch.int64)

        for batch_id,batch_center3ds in enumerate(center3ds): 
            mask[batch_id, :batch_center3ds.shape[0]] = mask_2d[batch_id]
        
        if int(mask.sum())>0:
            gt_bboxes = torch.cat(gt_bboxes).float()
            gt_bboxes_3d = torch.cat(gt_bboxes_3d)
            gt_labels = torch.cat(gt_labels)

            # original 2d bbox
            wh = gt_bboxes[:,2:]-gt_bboxes[:,:2]
            center2ds = (gt_bboxes[:,2:]+gt_bboxes[:,:2])/2

            center3ds = torch.cat(center3ds).float()
            center_heatmaps = center3ds.floor() if self.use_3d_center else center2ds.floor()
            radius = self.bbox_coder.gaussian_radius(wh.T,min_overlap=0.7)
            radius = torch.relu(radius.long())
            indices[mask] = (center_heatmaps[:, 1] * feat_w + center_heatmaps[:, 0]).long()     

            '''
            #!---------------tmp code--------------------
            # 50% prob to regress surround points
            final_h_coord = torch.clamp(center_heatmaps[:, 1]+(torch.rand(1).to(gt_labels.device)*2-1).round(), min=0, max=feat_h-1)
            final_w_coord = torch.clamp(center_heatmaps[:, 0]+(torch.rand(1).to(gt_labels.device)*2-1).round(), min=0, max=feat_w-1)
            indices[mask] = (final_h_coord * feat_w + final_w_coord).long()   
            #!-------------------------------------------
            '''


            size_2d[mask] = wh
            src_size_3d[mask] = gt_bboxes_3d[:,3:6]
            if self.bbox_code_size == 9: #nuscenes
                velo[mask] = gt_bboxes_3d[:, 7:9]  
            if self.pred_attrs:
                gt_attr_labels = torch.cat(gt_attr_labels)
                attributes[mask] = gt_attr_labels
            size_3d[mask] = src_size_3d[mask] - self.mean_size.to(gt_labels.device)[gt_labels].to(gt_labels.device)
            xyz[mask] = gt_bboxes_3d[:, :3]
            offset_2d[mask] = center2ds-center_heatmaps.float()
            offset_3d[mask] = center3ds-center_heatmaps.float()
            cls_ids[mask] = gt_labels
            obj_nums = torch.cat([mask[i].sum().unsqueeze(0) for i in range(len(img_metas))])

            calibs = torch.cat([torch.tensor(_['cam2img']).unsqueeze(0).repeat(obj_nums[i],1,1) for i,_ in enumerate(img_metas)]).to(gt_bboxes[0].device)
            trans_mats = torch.cat([torch.tensor(_['trans_mat']).unsqueeze(0).repeat(mask[i].sum(),1,1) for i,_ in enumerate(img_metas)]).to(gt_bboxes[0].device)
            heading_angles = self.bbox_coder.ry2alpha(gt_bboxes_3d[:, 6],gt_bboxes,calibs,trans_mats)
            heading_bin[mask.unsqueeze(-1)], heading_res[mask.unsqueeze(-1)] = self.bbox_coder.angle2class(heading_angles)

            obj_nums = torch.cat([obj_nums.new_zeros(1),obj_nums])
            for batch_id in range(1,obj_nums.shape[0]): 
                # index batch data
                left_index = obj_nums[:batch_id].sum()
                right_index = obj_nums[:batch_id+1].sum()
                if left_index == right_index: continue
                batch_gt_labels = gt_labels[left_index:right_index]
                batch_center_heatmaps = center_heatmaps[left_index:right_index]
                batch_radius = radius[left_index:right_index]
                for obj_id,(center,gt_label) in \
                        enumerate(zip(batch_center_heatmaps,batch_gt_labels)):
                    center_x_int, center_y_int = center.int()
                    gen_gaussian_target(heatmap[batch_id-1, gt_label],
                                        [center_x_int, center_y_int], batch_radius[obj_id])  

        avg_factor = max(1, heatmap.eq(1).sum())

        info = dict(
            avg_factor=avg_factor
        )

        target_labels = dict(
            mask=mask,
            cls_ids=cls_ids,
            xyz=xyz,
            size_2d=size_2d,
            heatmap=heatmap,
            offset_2d=offset_2d,
            indices=indices,
            size_3d=size_3d,
            src_size_3d=src_size_3d,
            offset_3d=offset_3d,
            heading_bin=heading_bin,
            heading_res=heading_res,
            velo=velo,
            attributes=attributes)
        return heatmap,target_labels,info

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             feat_maps,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             attr_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level.
                shape (num_gt, 4).
            bbox_preds (list[Tensor]): Box dims is a 4D-tensor, the channel
                number is bbox_code_size.
                shape (B, 7, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image.
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
                shape (num_gts, ).
            gt_bboxes_3d (list[:obj:`CameraInstance3DBoxes`]): 3D boxes ground
                truth. it is the flipped gt_bboxes
            gt_labels_3d (list[Tensor]): Same as gt_labels.
            centers2d (list[Tensor]): 2D centers on the image.
                shape (num_gts, 2).
            depths (list[Tensor]): Depth ground truth.
                shape (num_gts, ).
            attr_labels (list[Tensor]): Attributes indices of each box.
                In kitti it's None.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        center2d_heatmap_target, target_labels, infos = \
            self.get_targets(gt_bboxes, gt_labels, gt_bboxes_3d,
                             gt_labels_3d, attr_labels, centers2d,
                             feat_maps[0].shape,
                             img_metas[0]['pad_shape'],
                             img_metas) 

        center2d_heatmap, pred_reg, attr_preds = self(feat_maps, img_metas, target_labels)

        postive_bin_nums = target_labels['mask'].sum()
        if postive_bin_nums!=0:
            pred_reg[:2] = [
                transpose_and_gather_feat(_,target_labels['indices'])[target_labels['mask']] \
                    for _ in pred_reg[:2]]

            offset_2d, size_2d, depth, offset_3d, size_3d, heading = pred_reg

            avg_factor = infos['avg_factor']

            loss_cls = self.loss_cls(  
                center2d_heatmap, center2d_heatmap_target, avg_factor=avg_factor)

            loss_offset2d = self.loss_offset2d(
                offset_2d, 
                target_labels['offset_2d'][target_labels['mask']])

            loss_offset3d = self.loss_offset3d(
                offset_3d, 
                target_labels['offset_3d'][target_labels['mask']])
            
            if self.uncertainty_type == 'GeU':
                loss_size2d = self.loss_size2d(size_2d, target_labels['size_2d'][target_labels['mask']])       
            elif self.uncertainty_type == 'GeU++':    
                size_2d, size_2d_logstd = size_2d[:,:2], size_2d[:,2:]
                loss_size2d = self.loss_size2d(size_2d, target_labels['size_2d'][target_labels['mask']], size_2d_logstd)       

            size_3d, size_3d_logstd = size_3d[:,:3], size_3d[:,3:]
            loss_size3d = self.loss_size3d(size_3d, target_labels['src_size_3d'][target_labels['mask']], size_3d_logstd)    

            depth, depth_logstd = depth[:,:1], depth[:,1:]
            loss_depth = self.loss_depth(depth, target_labels['xyz'][target_labels['mask']][:,2:3], depth_logstd)

            heading_bin, heading_res = heading[:,:12], heading[:,12:]
            heading_cls_onehot = heading_bin.new_zeros(postive_bin_nums, 12)\
                .scatter_(dim=1, index=target_labels['heading_bin'][target_labels['mask']], value=1)

            loss_angle_bin = self.loss_angle_bin(
                heading_bin,target_labels['heading_bin'][target_labels['mask']].squeeze(1))

            loss_angle_res = self.loss_angle_res(
                (heading_res*heading_cls_onehot).sum(-1,keepdims=True),target_labels['heading_res'][target_labels['mask']])

            if self.pred_attrs:
                attr_preds = transpose_and_gather_feat(attr_preds,target_labels['indices'])[target_labels['mask']]
                loss_attr = self.loss_attr(attr_preds,target_labels['attributes'][target_labels['mask']])
            if self.pred_velo:
                loss_velo = self.loss_bbox(pred_reg[-1],target_labels['velo'][target_labels['mask']])
        else:
            offset_2d, size_2d, depth, offset_3d, size_3d, heading = pred_reg
            size_2d, size_2d_un = size_2d[:,:2], size_2d[:,2:]
            loss_depth = torch.zeros(1).cuda()
            loss_angle_bin = torch.zeros(1).cuda()
            loss_angle_res = torch.zeros(1).cuda()
            loss_offset3d = torch.zeros(1).cuda()
            loss_size3d = torch.zeros(1).cuda()
            loss_offset2d = torch.zeros(1).cuda()
            loss_cls = center2d_heatmap.softmax(1).mean()*0.0
            loss_size2d = torch.zeros(1).cuda()
            if self.pred_attrs: loss_attr = attr_preds.softmax(1).mean()*0.0
            if self.pred_velo: loss_velo = pred_reg[-1].mean()

        loss_dict = dict(
                         depth_loss=loss_depth,
                         heading_loss=loss_angle_bin+loss_angle_res,
                         offset2d_loss=loss_offset2d,
                         offset3d_loss=loss_offset3d,
                         seg_loss=loss_cls, 
                         size2d_loss=loss_size2d,
                         size3d_loss=loss_size3d,
                         )
        
        if hasattr(self,'heuristic_loss_scheme'):  #HTL for GUPNet
            self.heuristic_loss_scheme.update_losses(loss_dict)
            self.heuristic_loss_scheme.compute_weights(loss_dict)

        if self.pred_attrs: loss_dict['attr_loss'] = loss_attr
        if self.pred_velo: loss_dict['velo_loss'] = loss_velo
        return loss_dict

    def decode_bbox2d_maps(self, offset_2d, size_2d, trans_mats=None):
        with torch.no_grad():
            DEVICE, (BATCH_SIZE, _, HEIGHT, WIDE) = size_2d.device, size_2d.shape
            coord_maps = torch.stack([
                        torch.arange(WIDE,device=DEVICE).unsqueeze(0).repeat([HEIGHT,1]),
                        torch.arange(HEIGHT,device=DEVICE).unsqueeze(1).repeat([1,WIDE])]
                        ,0).repeat([BATCH_SIZE,1,1,1]).type(torch.float)  
        centers2d = coord_maps + offset_2d
        bbox2d_maps_featmaps = torch.cat([centers2d-size_2d[:,:2]/2,centers2d+size_2d[:,:2]/2],1)
        if trans_mats is not None:
            bbox2ds = torch.cat([bbox2d_maps_featmaps.permute(0,2,3,1).reshape(BATCH_SIZE,-1,2),
                                 bbox2d_maps_featmaps.new_ones(BATCH_SIZE,2*HEIGHT*WIDE,1)],dim=2)
            bbox2ds_in_img = torch.bmm(trans_mats.inverse(),bbox2ds.permute(0,2,1)).permute(0,2,1)
            bbox2d_maps = bbox2ds_in_img[:,:,:2].reshape(BATCH_SIZE,HEIGHT,WIDE,-1).permute(0,3,1,2)
        else:
            bbox2d_maps = None
        return bbox2d_maps_featmaps, bbox2d_maps
