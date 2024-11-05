# Copyright (c) OpenMMLab. All rights reserved.
import pdb
import numpy as np
import torch

from math import sqrt
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class GUPCoder(BaseBBoxCoder):
    """Bbox Coder for GUPNet.

    Args:
        base_dims (tuple[tuple[float]]): Dimension references [l, h, w]
            for decode box dimension for each category.
    """

    def __init__(self, num_heading_bin=12):
        super(GUPCoder, self).__init__()
        self.angle_per_class = 2 * np.pi / float(num_heading_bin)

    def encode(self):
        pass

    def decode(self,
               reg,
               points,
               labels,
               cam2imgs,
               trans_mats):
        """Decode regression into locations, dimensions, orientations.

        Args:
            reg (Tensor): Batch regression for each predict center2d point.
                shape: (batch * K (max_objs), C)
            points(Tensor): Batch projected bbox centers on image plane.
                shape: (batch * K (max_objs) , 2)
            labels (Tensor): Batch predict class label for each predict
                center2d point.
                shape: (batch, K (max_objs))
            cam2imgs (Tensor): Batch images' camera intrinsic matrix.
                shape: kitti (batch, 4, 4)  nuscenes (batch, 3, 3)
            trans_mats (Tensor): transformation matrix from original image
                to feature map.
                shape: (batch, 3, 3)

        Return:
            tuple(Tensor): The tuple has components below:
                - locations (Tensor): Centers of 3D boxes.
                    shape: (batch * K (max_objs), 3)
                - dimensions (Tensor): Dimensions of 3D boxes.
                    shape: (batch * K (max_objs), 3)
                - orientations (Tensor): Orientations of 3D 
                    boxes.
                    shape: (batch * K (max_objs), 1)
        """
        offset_2d, size_2d, depth, offset_3d, dimensions_3d, heading = reg
        bboxes_2d = self._decode_2d_box(points, offset_2d, size_2d, trans_mats)
        locations_3d = self._decode_location_3d (points, offset_3d,
                                               depth, cam2imgs, trans_mats)      
        orientations = self._decode_orientation(heading,bboxes_2d,cam2imgs)
        bboxes_3d = torch.cat((locations_3d, dimensions_3d, orientations), dim=2)

        return bboxes_2d, bboxes_3d
    def _decode_2d_box(self, points, offset_2d, size_2d, trans_mats):
        batch_size, max_objs, _ = offset_2d.shape
        trans_mats_inv = trans_mats.inverse()
        centers2d = points + offset_2d
        bbox2d = torch.cat([centers2d-size_2d/2,centers2d+size_2d/2],2)
        bbox2d = bbox2d.reshape([batch_size,2*max_objs,2])
        bbox2d_extend = torch.cat((bbox2d, 
                bbox2d.new_ones(batch_size, 2*max_objs, 1)),dim=2)
        bbox2d = torch.bmm(trans_mats_inv, bbox2d_extend.permute(0,2,1))\
                .permute(0,2,1)[:,:,:2].reshape([batch_size,max_objs,4])
        return bbox2d

    def _decode_location_3d(self, points, offsets_3d, depths, cam2imgs,
                         trans_mats):
        """Retrieve objects location in camera coordinate based on projected
        points.

        Args:
            points (Tensor): Projected points on feature map in (x, y)
                shape: (batch * K, 2)
            centers2d_offset (Tensor): Project points offset in
                (delta_x, delta_y). shape: (batch * K, 2)
            depths (Tensor): Object depth z.
                shape: (batch * K)
            cam2imgs (Tensor): Batch camera intrinsics matrix.
                shape: kitti (batch, 4, 4)  nuscenes (batch, 3, 3)
            trans_mats (Tensor): transformation matrix from original image
                to feature map.
                shape: (batch, 3, 3)
        """
        # number of points
        batch_size, max_objs, _ = offsets_3d.shape
        trans_mats_inv = trans_mats.inverse()
        cam2imgs_inv = cam2imgs.inverse()
        
        centers3d = points + offsets_3d
        centers3d_extend = torch.cat((centers3d, centers3d.new_ones(batch_size, max_objs, 1)),
                                     dim=2)
        
        # transform project points back on original image
        centers3d_img = torch.bmm(trans_mats_inv, centers3d_extend.permute(0,2,1)).permute(0,2,1)
        centers3d_img = centers3d_img * depths

        if cam2imgs.shape[1] == 4:
            centers3d_img = torch.cat(
                (centers3d_img, centers3d.new_ones(batch_size, max_objs, 1)), dim=-1)
        locations = torch.bmm(cam2imgs_inv, centers3d_img.permute(0,2,1)).permute(0,2,1)

        return locations[:, :, :3]

    def _decode_orientation(self, heading, bbox2d, cam2imgs):
        """Retrieve object orientation.
        Return:
            Tensor: yaw(Orientation). Notice that the yaw's
                range is [-np.pi, np.pi].
                shape：(N, 1）
        """
        batch_size, max_objs, _ = heading.shape
        res = heading.new_zeros(batch_size, max_objs)
        heading_bin, heading_res = heading[:,:,0:12], heading[:,:,12:24]
        cls = heading_bin.argmax(-1)
        for batch_id, (batch_res, batch_cls) in \
            enumerate(zip(heading_res, cls)):
            cls_onehot = batch_cls.new_zeros(batch_res.shape)\
                .scatter_(dim=1, index=batch_cls.view(-1,1), value=1)
            res[batch_id] = (batch_res*cls_onehot).sum(-1)
        alpha = self.class2angle(cls, res, to_label_format=True)
        yaws = self.alpha2ry(alpha, bbox2d, cam2imgs)
        return yaws.unsqueeze(-1)

    def decode_uncertainty(self,
               reg,
               reg_logstd,
               points,
               labels,
               cam2imgs,
               trans_mats):
        offset_2d, size_2d, depth, offset_3d, size_3d, heading = reg
        offset_2d_logstd, size_2d_logstd, depth_logstd, offset_3d_logstd, size_3d_logstd, heading_slogtd = reg_logstd
        locations_3d_logstd = self._decode_location_3d_un(points, offset_3d,
                                               depth_logstd, cam2imgs, trans_mats)
        dimensions_3d_logstd = size_3d_logstd
        orientations_logstd = heading.new_zeros([heading.shape[0],heading.shape[1],1])
        bboxes_3d_logstd = torch.cat((locations_3d_logstd, dimensions_3d_logstd, orientations_logstd), dim=2)
        return bboxes_3d_logstd
        
    def _decode_location_3d_un(self, points, offsets_3d, depth_logstd, cam2imgs,
                         trans_mats):
        """Retrieve objects location in camera coordinate based on projected
        points.

        Args:
            points (Tensor): Projected points on feature map in (x, y)
                shape: (batch * K, 2)
            centers2d_offset (Tensor): Project points offset in
                (delta_x, delta_y). shape: (batch * K, 2)
            depths (Tensor): Object depth z.
                shape: (batch * K)
            cam2imgs (Tensor): Batch camera intrinsics matrix.
                shape: kitti (batch, 4, 4)  nuscenes (batch, 3, 3)
            trans_mats (Tensor): transformation matrix from original image
                to feature map.
                shape: (batch, 3, 3)
        """
        # number of points
        batch_size, max_objs, _ = offsets_3d.shape
        trans_mats_inv = trans_mats.inverse()
        cam2imgs_inv = cam2imgs.inverse()
        
        centers3d = points + offsets_3d
        centers3d_extend = torch.cat((centers3d, centers3d.new_ones(batch_size, max_objs, 1)),
                                     dim=2)
        
        # transform project points back on original image
        centers3d_img = torch.bmm(trans_mats_inv, centers3d_extend.permute(0,2,1)).permute(0,2,1)

        centers3d_img_logvar = torch.log(centers3d_img**2+1e-12) + 2*depth_logstd

        if cam2imgs.shape[1] == 4:
            cam2imgs_inv = cam2imgs_inv[:,:3,:3]
        locations_logvar = (torch.log(cam2imgs_inv**2+1e-12).unsqueeze(2) + centers3d_img_logvar.unsqueeze(1)).logsumexp(3).permute(0,2,1)
        #locations_logstd = torch.bmm(cam2imgs_inv**2, centers3d_img_logvar.exp().permute(0,2,1)).permute(0,2,1).log()
        return 0.5*locations_logvar

    def gaussian_radius(self, det_size, min_overlap): 
        width, height = det_size
        wh_sum = width + height
        wh_mul = width * height

        a1 = 1
        b1 = wh_sum
        c1 = wh_mul * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * wh_sum
        c2 = (1 - min_overlap) * wh_mul
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * wh_sum
        c3 = (min_overlap - 1) * wh_mul
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return torch.stack([r1,r2,r3]).min(0)[0]

    def alpha2ry(self, alpha, bbox2d, calib):
        u = (bbox2d[:,:,0]+bbox2d[:,:,2])/2
        ry = alpha + torch.atan2(u - calib[:,0,2].unsqueeze(-1), calib[:,0,0].unsqueeze(-1))
        ry[ry > np.pi] -= 2 * np.pi
        ry[ry < -np.pi] += 2 * np.pi
        return ry

    def ry2alpha(self, ry, gt_bbox_2d, calibs, trans_mats):
        trans_mats = trans_mats.repeat(2,1,1)
        bbox2d_corners = torch.cat([torch.cat([gt_bbox_2d[:,:2],gt_bbox_2d[:,2:]]),
            gt_bbox_2d.new_ones([gt_bbox_2d.shape[0]*2,1])],1).unsqueeze(-1)
        ori_gt_bbox_2d = torch.bmm(torch.inverse(trans_mats),
            bbox2d_corners)[:,:2][:,:,0]    
        u = ori_gt_bbox_2d[:,0].view(2,-1).mean(0)
        alpha = ry - torch.atan2(u - calibs[:,0,2], calibs[:,0,0])
        alpha[alpha > np.pi]-= 2 * np.pi
        alpha[alpha < -np.pi]+= 2 * np.pi
        return alpha

    def angle2class(self,angle):
        ''' Convert continuous angle to discrete class and residual. '''
        angle = angle % (2 * np.pi)
        assert False not in (angle >= 0)*(angle <= 2 * np.pi)
        shifted_angle = (angle + self.angle_per_class / 2) % (2 * np.pi)
        class_id = (shifted_angle / self.angle_per_class).long()
        residual_angle = shifted_angle - (class_id * self.angle_per_class + self.angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, cls, residual, to_label_format=False):
        ''' Inverse function to angle2class. '''
        angle_center = cls * self.angle_per_class
        angle = angle_center + residual
        if to_label_format:
            angle[angle > np.pi] = angle[angle > np.pi] - 2 * np.pi
        return angle