import math
import numpy as np

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.models.ostrack.utils import combine_tokens
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name, threshold):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.update_template = False
        self.threshold = threshold
        self.templates = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict, id = None):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        if id:
            self.frame_id = id
        self.state = info['init_bbox']
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
            
    def initialize_multiple(self, image, info: list, id = None):
        # forward the template once
        self.templates = None
        self.box_mask_zs = None
        self.states = None
        # combine_tokens(self.box_mask_z1, self.box_mask_z1, mode='direct')
        for temp_bbox in info:
            z_patch_arr, resize_factor, z_amask_arr = sample_target(image, temp_bbox, self.params.template_factor,
                                                        output_sz=self.params.template_size)
            self.z_patch_arr = z_patch_arr
            template = self.preprocessor.process(z_patch_arr, z_amask_arr)
            with torch.no_grad():
                # print(template.tensors.shape)
                if self.templates is not None:
                    self.templates = combine_tokens(self.templates, template.tensors, mode='direct')
                else:
                    self.templates = template.tensors

            if self.cfg.MODEL.BACKBONE.CE_LOC:
                template_bbox = self.transform_bbox_to_crop(temp_bbox, resize_factor,
                                                            template.tensors.device).squeeze(1)
                if self.box_mask_zs is not None:
                    temp = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
                    self.box_mask_zs = combine_tokens(self.box_mask_zs, temp, mode='direct')
                else:
                    self.box_mask_zs = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

            # save states
            if id:
                self.frame_id = id
            if self.states is not None:
                self.states = np.vstack((self.states, temp_bbox)).reshape(-1, 4)
            else:
                self.states = np.reshape(temp_bbox, (-1, 4))


    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        # no hann windows
        with torch.no_grad():
            no_hanning_boxes = self.network.box_head.cal_bbox(out_dict['score_map'], out_dict['size_map'], out_dict['offset_map'])
            no_hanning_boxes = no_hanning_boxes.view(-1, 4)
            no_hanning_box = (no_hanning_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            no_hanning_result = clip_box(self.map_box_back(no_hanning_box, resize_factor), H, W, margin=10)
        # if no_hanning_box != pred_box or self.frame_id == 1261:
        #     print("Frame_id:", self.frame_id)
        #     print("no hanning:", no_hanning_result)
        #     print("hanning:", self.state)
        
        # calculate distance of hann and no hanning window
        self.update_template = False
        dist = self.calc_center_dist(self.state, no_hanning_result)
        norm_dist = self.calc_center_dist(self.state, no_hanning_result, True)

        if norm_dist > self.threshold:
            self.update_template = True
            # print("Frame_id:", self.frame_id)
            # print("dist:", dist, "norm_dist:", norm_dist)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "update_template": self.update_template}
        else:
            return {"target_bbox": self.state, "update_template": self.update_template}
    
    def track_multiple(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        
        self.searchs = None
        self.resize_factors = []
        # print("state:", self.states.shape)
        for state in self.states:
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
            self.resize_factors.append(resize_factor)
            search = self.preprocessor.process(x_patch_arr, x_amask_arr)
            if self.searchs is not None:
                    self.searchs = combine_tokens(self.searchs, search.tensors, mode='direct')
            else:
                self.searchs = search.tensors

        with torch.no_grad():
            # x_dict = search
            templates = torch.reshape(self.templates, (-1, 3, self.templates.shape[2], self.templates.shape[3]))
            # print(templates.shape)
            searchs = torch.reshape(self.searchs, (-1, 3, self.searchs.shape[2], self.searchs.shape[3]))
            # print(searchs.shape)
            B = searchs.shape[0]
            # print(self.box_mask_zs.shape)
            box_mask_zs = torch.reshape(self.box_mask_zs, (B, -1))
            # print(box_mask_zs.shape)

            # box_mask_zs = torch.cat(self.box_mask_zs)
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=templates, search=searchs, ce_template_mask=box_mask_zs)
        # print("out_dict:", out_dict['score_map'].shape) # (B, 1, 24, 24)
        # print("out_dict:", out_dict['size_map'].shape) # (B, 2, 24, 24)
        # print("out_dict:", out_dict['offset_map'].shape) # (B, 2, 24, 24)
        # add hann windows
        for idx in range(B):
            pred_score_map = out_dict['score_map'][idx].unsqueeze(0)
            # print(pred_score_map.shape)
            # print(out_dict['size_map'][idx].unsqueeze(0).shape)
            # print(out_dict['offset_map'][idx].unsqueeze(0).shape)
            response = self.output_window * pred_score_map
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'][idx].unsqueeze(0), out_dict['offset_map'][idx].unsqueeze(0))
            pred_boxes = pred_boxes.view(-1, 4)
            # print(pred_boxes.shape)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / self.resize_factors[idx]).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.states[idx] = clip_box(self.map_box_back_multiple(pred_box, self.resize_factors[idx], self.states[idx]), H, W, margin=10)
            
        # # no hann windows
        # with torch.no_grad():
        #     no_hanning_boxes = self.network.box_head.cal_bbox(out_dict['score_map'], out_dict['size_map'], out_dict['offset_map'])
        #     no_hanning_boxes = no_hanning_boxes.view(-1, 4)
        #     no_hanning_box = (no_hanning_boxes.mean(
        #         dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        #     no_hanning_result = clip_box(self.map_box_back(no_hanning_box, resize_factor), H, W, margin=10)
        
        # # calculate distance of hann and no hanning window
        # self.update_template = False
        # dist = self.calc_center_dist(self.state, no_hanning_result)
        # norm_dist = self.calc_center_dist(self.state, no_hanning_result, True)

        # if norm_dist > self.threshold:
        #     self.update_template = True
        #     # print("Frame_id:", self.frame_id)
        #     # print("dist:", dist, "norm_dist:", norm_dist)
        return {"target_bbox": self.states}#, "update_template": self.update_template}


    def calc_center_dist(self, pred_bb, anno_bb, normalized=False):
        pred_bb = np.array(pred_bb)
        anno_bb = np.array(anno_bb)
        pred_center = pred_bb[:2] + 0.5 * (pred_bb[2:] - 1.0)
        anno_center = anno_bb[:2] + 0.5 * (anno_bb[2:] - 1.0)

        if normalized:
            pred_center = pred_center / anno_bb[2:]
            anno_center = anno_center / anno_bb[2:]

        center_dist = np.sqrt(((pred_center - anno_center)**2).sum())
        return center_dist

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_multiple(self, pred_box: list, resize_factor: float, state):
        cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack
