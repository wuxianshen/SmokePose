"""
@File  : skp_eval_bottom_up.py
@Author: tao.jing
@Date  : 2022/4/12
@Desc  :
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import paddle
import numpy as np

from ppdet.smoke_kp.skp_utils import skp_get_infer_results, skp_cocoapi_eval
from ppdet.data.source.category import get_categories

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'SKPMetric',
    'SKPCOCOMetric',
]

SKP_COCO_SIGMAS = np.array([
    .26, .25, ]) / 10.0

CROWD_SIGMAS = np.array(
    [.79, .79, ]) / 10.0


class SKPMetric(paddle.metric.Metric):
    def name(self):
        return self.__class__.__name__

    def reset(self):
        pass

    def accumulate(self):
        pass

    # paddle.metric.Metric defined :metch:`update`, :meth:`accumulate`
    # :metch:`reset`, in ppdet, we also need following 2 methods:

    # abstract method for logging metric results
    def log(self):
        pass

    # abstract method for getting metric results
    def get_results(self):
        pass


class SKPCOCOMetric(SKPMetric):
    def __init__(self, anno_file, **kwargs):
        super(SKPCOCOMetric, self).__init__()
        assert os.path.isfile(anno_file), \
                "anno_file {} not a file".format(anno_file)
        self.anno_file = anno_file
        self.clsid2catid = kwargs.get('clsid2catid', None)
        if self.clsid2catid is None:
            self.clsid2catid, _ = get_categories('COCO', anno_file)
        self.classwise = kwargs.get('classwise', False)
        self.output_eval = kwargs.get('output_eval', None)
        # TODO: bias should be unified
        self.bias = kwargs.get('bias', 0)
        self.save_prediction_only = kwargs.get('save_prediction_only', False)
        self.iou_type = kwargs.get('IouType', 'bbox')
        self.reset()

    def reset(self):
        # only bbox and mask evaluation support currently
        self.results = {'bbox': [], 'mask': [], 'segm': [], 'keypoint': []}
        self.eval_results = {}

    def update(self, inputs, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        im_id = inputs['im_id']
        outs['im_id'] = im_id.numpy() if isinstance(im_id,
                                                    paddle.Tensor) else im_id

        infer_results = skp_get_infer_results(
            outs, self.clsid2catid, bias=self.bias)
        self.results['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []
        self.results['mask'] += infer_results[
            'mask'] if 'mask' in infer_results else []
        self.results['segm'] += infer_results[
            'segm'] if 'segm' in infer_results else []
        self.results['keypoint'] += infer_results[
            'keypoint'] if 'keypoint' in infer_results else []

    def accumulate(self):
        if len(self.results['bbox']) > 0:
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['bbox'], f)
                logger.info('The bbox result is saved to bbox.json.')

            if self.save_prediction_only:
                logger.info('The bbox result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                bbox_stats = skp_cocoapi_eval(
                    output,
                    'bbox',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['bbox'] = bbox_stats
                sys.stdout.flush()

        if len(self.results['mask']) > 0:
            output = "mask.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['mask'], f)
                logger.info('The mask result is saved to mask.json.')

            if self.save_prediction_only:
                logger.info('The mask result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                seg_stats = skp_cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

        if len(self.results['segm']) > 0:
            output = "segm.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['segm'], f)
                logger.info('The segm result is saved to segm.json.')

            if self.save_prediction_only:
                logger.info('The segm result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                seg_stats = skp_cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

        if len(self.results['keypoint']) > 0:
            output = "keypoint.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['keypoint'], f)
                logger.info('The keypoint result is saved to keypoint.json.')

            if self.save_prediction_only:
                logger.info('The keypoint result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                style = 'keypoints'
                use_area = True
                sigmas = SKP_COCO_SIGMAS
                if self.iou_type == 'keypoints_crowd':
                    style = 'keypoints_crowd'
                    use_area = False
                    sigmas = CROWD_SIGMAS
                keypoint_stats = skp_cocoapi_eval(
                    output,
                    style,
                    anno_file=self.anno_file,
                    classwise=self.classwise,
                    sigmas=sigmas,
                    use_area=use_area)
                self.eval_results['keypoint'] = keypoint_stats
                sys.stdout.flush()

    def log(self):
        pass

    def get_results(self):
        return self.eval_results