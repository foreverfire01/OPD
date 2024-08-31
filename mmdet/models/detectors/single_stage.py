# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import mmdet

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

import numpy as np
from mmdet.models.losses.focal_loss import FocalLoss
import mmdet.models.global_variable as global_variable
from mmdet.utils import get_root_logger

import os
from mmdet.models.detectors.zdetector_util import DetectorUtil

import datetime



device = 'cuda:0'      # DOG

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_opd = FocalLoss(use_sigmoid=True,            # DOG
                                  gamma=2,
                                  alpha=0.75,
                                  loss_weight=2.0)
        self.logger = get_root_logger()

        self.util = DetectorUtil(self.loss_opd)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    # def forward_train(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   gt_bboxes_ignore=None):
    #     """
    #     Args:
    #         img (Tensor): Input images of shape (N, C, H, W).
    #             Typically these should be mean centered and std scaled.
    #         img_metas (list[dict]): A List of image info dict where each dict
    #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
    #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
    #             For details on the values of these keys see
    #             :class:`mmdet.datasets.pipelines.Collect`.
    #         gt_bboxes (list[Tensor]): Each item are the truth boxes for each
    #             image in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels (list[Tensor]): Class indices corresponding to each box
    #         gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
    #             boxes can be ignored when computing the loss.

    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     super(SingleStageDetector, self).forward_train(img, img_metas)
    #     x = self.extract_feat(img)
    #     losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
    #                                           gt_labels, gt_bboxes_ignore)
    #     return losses

    # def simple_test(self, img, img_metas, rescale=False):
    #     """Test function without test-time augmentation.

    #     Args:
    #         img (torch.Tensor): Images with shape (N, C, H, W).
    #         img_metas (list[dict]): List of image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.

    #     Returns:
    #         list[list[np.ndarray]]: BBox results of each image and classes.
    #             The outer list corresponds to each image. The inner list
    #             corresponds to each class.
    #     """
    #     feat = self.extract_feat(img)
    #     results_list = self.bbox_head.simple_test(
    #         feat, img_metas, rescale=rescale)
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in results_list
    #     ]
    #     return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels







# -----------------------------------------------------------------------------------------------------------

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        
        super(SingleStageDetector, self).forward_train(img, img_metas)

        # --------------------------------------------------------------------------------
        if(type(self.backbone) == mmdet.models.OPDSwinTransformer):
            x, outopdmap = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
            
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

            batch_size = img.shape[0]

            # losses = {}
            losses['loss_opd1'] = 2 * self.util.cal_layer_loss_statistic(img, gt_bboxes, outopdmap[0], '1', is_calloss = True, is_adjust = True)
            losses['loss_opd2'] = 2 * self.util.cal_layer_loss_statistic(img, gt_bboxes, outopdmap[1], '2', is_calloss = True, is_adjust = True)
            losses['loss_opd3'] = self.util.cal_layer_loss_statistic(img, gt_bboxes, outopdmap[2], '3', is_calloss = True, is_adjust = True)
            losses['loss_opd4'] = self.util.cal_layer_loss_statistic(img, gt_bboxes, outopdmap[3], '4', is_calloss = True, is_adjust = False)

            # self.sutil.tatistic_all4(batch_size, outopdmap[0].shape[1], 23220, is_print = False)
            picsum = global_variable.get_value('picsum') 
            if picsum == 23220:  # ASDD:full-23220  1w-2632
                global_variable.reset()  
                                                                                   
        # --------------------------------------------------------------------------------
        else:
            x = self.extract_feat(img)
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

        return losses


    # def simple_test(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   rescale=False):
    #     '''
    #         传入真值框对比，注意修改相应配置文件 keys=['img', 'gt_bboxes', 'gt_labels']
    #     '''
    #     # --------------------------------------------------------------------------------
    #     if(type(self.backbone) == mmdet.models.OPDSwinTransformer):
    #         feat, outopdmap = self.backbone(img)
    #         if self.with_neck:
    #             feat = self.neck(feat)
            
    #         # --------------------------------------最大值保存---------------
    #         time = datetime.datetime.now().strftime("%Y%m%d")
    #         if gt_bboxes[0].shape[1] > 0:
    #             save_file = time + "_result_ship_train_retina_asdd" + ".csv"
    #         else:
    #             save_file = time + "_result_noship_train_retina_asdd" + ".csv"
            
    #         file_path = img_metas[0]['filename']
    #         file_name = os.path.basename(file_path)[0: -4]

    #         maxvalue1 = torch.max(outopdmap[0][:, :, :, 0]).sigmoid().item()
    #         maxvalue2 = torch.max(outopdmap[1][:, :, :, 0]).sigmoid().item()
    #         maxvalue3 = torch.max(outopdmap[2][:, :, :, 0]).sigmoid().item()
    #         maxvalue4 = torch.max(outopdmap[3][:, :, :, 0]).sigmoid().item()


    #         with open(save_file, "a") as f:
    #             f.writelines("\n%s, %.3f, %.3f, %.3f, %.3f" 
    #                          %(file_name, maxvalue1, maxvalue2, maxvalue3, maxvalue4))
    #         # -----------------------------------------------------

    #         # --------------------------------------AP/AR统计---------------
    #         # self.draw_heatmap(img, img_metas, feat, outopdmap)
    #         # batch_size = img.shape[0] 

    #         # self.util.cal_layer_loss_statistic(img, gt_bboxes[0], outopdmap[0], '1', is_calloss = False, is_adjust = False)
    #         # self.util.cal_layer_loss_statistic(img, gt_bboxes[0], outopdmap[1], '2', is_calloss = False, is_adjust = False)
    #         # self.util.cal_layer_loss_statistic(img, gt_bboxes[0], outopdmap[2], '3', is_calloss = False, is_adjust = False)
    #         # self.util.cal_layer_loss_statistic(img, gt_bboxes[0], outopdmap[3], '4', is_calloss = False, is_adjust = False)

    #         # self.util.statistic_all4(batch_size, outopdmap[0].shape[1], 10407, is_print = False) #  ASDD:val: test:10407   DOTA: val1047
    #         # -----------------------------------------------------

    #     else:
    #         feat = self.extract_feat(img)


    #     results_list = self.bbox_head.simple_test(
    #         feat, img_metas, rescale=rescale)
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in results_list
    #     ]

    #     # noship = [[np.empty((0,5), dtype='float32')]] 
    #     # return noship
    #     return bbox_results



    
            
    def simple_test(self, img, img_metas, rescale=False):
        '''
            不传真值框，注意修改相应配置文件 keys=['img']
        '''

        if(type(self.backbone) == mmdet.models.OPDSwinTransformer):
            feat, outopdmap = self.backbone(img)
            # self.util.draw_heatmap(img, img_metas, feat, outopdmap)

            if feat == None:
                noship = [[np.empty((0,5), dtype='float32')]]
                return noship
            
            elif self.with_neck:
                feat = self.neck(feat)

        else:
            feat = self.extract_feat(img)

        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]

        return bbox_results


