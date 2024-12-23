"""
Filename: deformgs.py

Author: Ziyu Chen (ziyu.sjtu@gmail.com)

Description:
Unofficial implementation of Defromable-GS based on the work by Ziyi Yang et al.
Original work by Ziyi Yang et al.

Original paper: https://arxiv.org/abs/2309.13101
"""

from typing import Union, Tuple, Dict, List
import logging
import torch
import numpy as np
from torch.nn import Parameter

from models.gaussians.basics import *
from models.gaussians.vanilla import VanillaGaussians
from models.modules import DeformNetwork,DeformNetwork_MIX,DeformNetwork_MIX_V2,DeformNetwork_MIX_V3,DeformNetwork_MIX_V4
from pyquaternion import Quaternion

logger = logging.getLogger()

def contract(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[str, int] = None,
) -> torch.Tensor:
    """
    Contract the input tensor to the unit cube using piecewise projective function.
    """
    # similar to the one in MeRF paper
    aabb_min, aabb_max = aabb[0, :], aabb[1, :]
    x = (x - aabb_min) / (aabb_max - aabb_min)  # 0~1
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    x = torch.where(mag < 1, x, (2 - 1 / mag) * (x / mag))
    x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    return x

def contract_c2w(
    x: torch.Tensor,
    aabb: torch.Tensor,
    
) -> torch.Tensor:
    """
    Contract the input tensor to the unit cube using piecewise projective function.
    """
    # similar to the one in MeRF paper
    trans = x[:3,3].reshape(-1)
    rotations = x[:3,:3].detach().cpu().numpy()
    q = Quaternion(matrix=rotations)
    q = torch.from_numpy(np.array([q.w,q.x,q.y,q.z]).astype(np.float32)).cuda()
    
    aabb_min, aabb_max = aabb[0, :], aabb[1, :]
    aabb_min = aabb_min.clone()
    aabb_min[0] = 0
    trans = (trans - aabb_min) / (aabb_max - aabb_min)  # 0~1

    q = (q+1)/2
    
    x = torch.cat([q,trans])
    return x

def contract_c2w_tran(
    x: torch.Tensor,
    aabb: torch.Tensor,
    
) -> torch.Tensor:
    """
    Contract the input tensor to the unit cube using piecewise projective function.
    """
    # similar to the one in MeRF paper
    trans = x[:3,3].reshape(-1)
    # rotations = x[:3,:3].detach().cpu().numpy()
    # q = Quaternion(matrix=rotations)
    # q = torch.from_numpy(np.array([q.w,q.x,q.y,q.z]).astype(np.float32)).cuda()
    aabb_min = torch.Tensor([1.5502,-0.0275,2.1139]).cuda()
    # aabb_min, aabb_max = aabb[0, :], aabb[1, :]
    # aabb_min = aabb_min.clone()
    # aabb_min[0] = 0
    length = torch.Tensor([100,10,5]).cuda()
    trans = (trans - aabb_min) / length  # 0~1

    # q = (q+1)/2
    return trans
    # x = trans
    # return x

def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper

class DeformableGaussians(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.deform_network = DeformNetwork(
            input_ch=3, **self.networks_cfg
        ).to(self.device)
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        self.delta_xyz_rescale = self.ctrl_cfg.get("delta_xyz_rescale", True)
    
    @property
    def defrom_gs(self):
        if self.step < self.ctrl_cfg.coarse_train_interval:
            return False
        else:
            return True
    
    def set_bbox(self, bbox: torch.Tensor):
        self.bbox = bbox.reshape(2, 3)
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id
   
    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
        self.time_interval = 1 / len(normalized_timestamps)
        
    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        super().create_from_pcd(init_means, init_colors)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        return param_groups
    
    def get_deformation(self, canonical_means) -> Tuple:
        """
        get the deformation of the nonrigid instances
        """        
        t = self.normalized_timestamps[self.cur_frame]
        t = t.unsqueeze(0).repeat(self.num_points, 1)
        normed_canonical_means = contract(canonical_means, self.bbox) # range from 0 to 1
        
        ast_noise = torch.randn(1, 1, device=self.device).expand(self.num_points, -1) * self.time_interval * self.smooth_term(self.step)
        
        delta_xyz, delta_quat, delta_scale = self.deform_network(normed_canonical_means.data, t + ast_noise)
        return delta_xyz, delta_quat, delta_scale
    
    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        delta_xyz, delta_quat, delta_scale = None, None, None
        if self.defrom_gs:
            delta_xyz, delta_quat, delta_scale = self.get_deformation(self._means)
            if self.delta_xyz_rescale:
                delta_xyz = delta_xyz * self.scene_scale

        if delta_xyz is not None:
            world_means = self._means + delta_xyz
        else:
            world_means = self._means
        
        if delta_quat is not None:
            world_quats = self.get_quats + delta_quat
        else:
            world_quats = self.get_quats
        
        if delta_scale is not None:
            activated_scales = torch.exp(self._scales + delta_scale)
        else:
            activated_scales = torch.exp(self._scales)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict

class DeformableGaussians_MIX(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.deform_network = DeformNetwork_MIX(
            input_ch=3, **self.networks_cfg
        ).to(self.device)
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        self.delta_xyz_rescale = self.ctrl_cfg.get("delta_xyz_rescale", True)

        # self.deform_network_gen = DeformNetwork(
        #     input_ch=3, **self.networks_cfg
        # ).to(self.device)
        # self.smooth_term_gen = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        # self.delta_xyz_rescale = self.ctrl_cfg.get("delta_xyz_rescale", True)
    
    @property
    def defrom_gs(self):
        if self.step < self.ctrl_cfg.coarse_train_interval:
            return False
        else:
            return True
    
    def set_bbox(self, bbox: torch.Tensor):
        self.bbox = bbox.reshape(2, 3)
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id
   
    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
        self.time_interval = 1 / len(normalized_timestamps)
        
    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        super().create_from_pcd(init_means, init_colors)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        return param_groups
    
    def get_deformation(self, canonical_means,canonical_c2ws) -> Tuple:
        """
        get the deformation of the nonrigid instances
        """        
        t = self.normalized_timestamps[self.cur_frame]
        t = t.unsqueeze(0).repeat(self.num_points, 1)
        normed_canonical_means = contract(canonical_means, self.bbox) # range from 0 to 1
        ast_noise = torch.randn(1, 1, device=self.device).expand(self.num_points, -1) * self.time_interval * self.smooth_term(self.step)

        normed_canonical_c2ws  = contract_c2w(canonical_c2ws,self.bbox)
        normed_canonical_c2ws = normed_canonical_c2ws.unsqueeze(0).repeat(self.num_points, 1)

        delta_xyz, delta_quat, delta_scale = self.deform_network(normed_canonical_means.data, t + ast_noise,normed_canonical_c2ws)
        return delta_xyz, delta_quat, delta_scale
    
    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        delta_xyz, delta_quat, delta_scale = None, None, None
        if self.defrom_gs:
            delta_xyz, delta_quat, delta_scale = self.get_deformation(self._means,cam.camtoworlds)
            if self.delta_xyz_rescale:
                delta_xyz = delta_xyz * self.scene_scale

        if delta_xyz is not None:
            world_means = self._means + delta_xyz
        else:
            world_means = self._means
        
        if delta_quat is not None:
            world_quats = self.get_quats + delta_quat
        else:
            world_quats = self.get_quats
        
        if delta_scale is not None:
            activated_scales = torch.exp(self._scales + delta_scale)
        else:
            activated_scales = torch.exp(self._scales)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict

class DeformableGaussians_MIX_V2(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.deform_network = DeformNetwork(
            input_ch=3, **self.networks_cfg
        ).to(self.device)
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        self.delta_xyz_rescale = self.ctrl_cfg.get("delta_xyz_rescale", True)

        self.deform_network_ref = DeformNetwork_MIX_V2(
            input_ch=3, **self.networks_cfg
        ).to(self.device)
        # self.smooth_term_gen = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        # self.delta_xyz_rescale = self.ctrl_cfg.get("delta_xyz_rescale", True)
    
    @property
    def defrom_gs(self):
        if self.step < self.ctrl_cfg.coarse_train_interval:
            return False
        else:
            return True
    
    def set_bbox(self, bbox: torch.Tensor):
        self.bbox = bbox.reshape(2, 3)
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id
   
    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
        self.time_interval = 1 / len(normalized_timestamps)
        
    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        super().create_from_pcd(init_means, init_colors)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        # param_groups[self.class_prefix+"deform_network_ref"] = list(self.deform_network_ref.parameters())
        return param_groups
    
    def get_param_groups_ref(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        param_groups[self.class_prefix+"deform_network_ref"] = list(self.deform_network_ref.parameters())
        return param_groups
    
    def get_deformation(self, canonical_means,canonical_c2ws,ref_flag) -> Tuple:
        """
        get the deformation of the nonrigid instances
        """        
        t = self.normalized_timestamps[self.cur_frame]
        t = t.unsqueeze(0).repeat(self.num_points, 1)
        normed_canonical_means = contract(canonical_means, self.bbox) # range from 0 to 1
        

        if ref_flag:
            ast_noise = torch.randn(1, 1, device=self.device).expand(self.num_points, -1) * self.time_interval * self.smooth_term(50000)
            with torch.no_grad():
                delta_xyz, delta_quat, delta_scale = self.deform_network(normed_canonical_means.data, t + ast_noise)
            normed_canonical_c2ws  = contract_c2w(canonical_c2ws,self.bbox)
            normed_canonical_c2ws = normed_canonical_c2ws.unsqueeze(0).repeat(self.num_points, 1)
            delta_xyz_ref, delta_quat_ref, delta_scale_ref = self.deform_network_ref(normed_canonical_means.data,normed_canonical_c2ws.data)
            delta_xyz += delta_xyz_ref
            delta_quat += delta_quat_ref
            delta_scale += delta_scale_ref
        else:
            ast_noise = torch.randn(1, 1, device=self.device).expand(self.num_points, -1) * self.time_interval * self.smooth_term(self.step)
            delta_xyz, delta_quat, delta_scale = self.deform_network(normed_canonical_means.data, t + ast_noise)
        
        return delta_xyz, delta_quat, delta_scale
    
    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        delta_xyz, delta_quat, delta_scale = None, None, None
        if self.defrom_gs:
            delta_xyz, delta_quat, delta_scale = self.get_deformation(self._means,cam.camtoworlds,cam.ref_flag)
            if self.delta_xyz_rescale:
                delta_xyz = delta_xyz * self.scene_scale
        
        if delta_xyz is not None:
            world_means = self._means + delta_xyz
        else:
            world_means = self._means
        
        if delta_quat is not None:
            world_quats = self.get_quats + delta_quat
        else:
            world_quats = self.get_quats
        
        if delta_scale is not None:
            activated_scales = torch.exp(self._scales + delta_scale)
        else:
            activated_scales = torch.exp(self._scales)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict

class DeformableGaussians_MIX_V3(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.deform_network = DeformNetwork(
            input_ch=3, **self.networks_cfg
        ).to(self.device)
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        self.delta_xyz_rescale = self.ctrl_cfg.get("delta_xyz_rescale", True)

        self.deform_network_ref = DeformNetwork_MIX_V3(
            input_ch=3, **self.networks_cfg
        ).to(self.device)
        # self.smooth_term_gen = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        # self.delta_xyz_rescale = self.ctrl_cfg.get("delta_xyz_rescale", True)
    
    @property
    def defrom_gs(self):
        if self.step < self.ctrl_cfg.coarse_train_interval:
            return False
        else:
            return True
    
    def set_bbox(self, bbox: torch.Tensor):
        self.bbox = bbox.reshape(2, 3)
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id
   
    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
        self.time_interval = 1 / len(normalized_timestamps)
        
    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        super().create_from_pcd(init_means, init_colors)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        # param_groups[self.class_prefix+"deform_network_ref"] = list(self.deform_network_ref.parameters())
        return param_groups
    
    def get_param_groups_ref(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        param_groups[self.class_prefix+"deform_network_ref"] = list(self.deform_network_ref.parameters())
        return param_groups
    
    def get_deformation(self, canonical_means,canonical_c2ws,ref_flag) -> Tuple:
        """
        get the deformation of the nonrigid instances
        """        
        t = self.normalized_timestamps[self.cur_frame]
        t = t.unsqueeze(0).repeat(self.num_points, 1)
        normed_canonical_means = contract(canonical_means, self.bbox) # range from 0 to 1
        

        if ref_flag:
            ast_noise = torch.randn(1, 1, device=self.device).expand(self.num_points, -1) * self.time_interval * self.smooth_term(50000)
            with torch.no_grad():
                delta_xyz, delta_quat, delta_scale = self.deform_network(normed_canonical_means.data, t + ast_noise)
            normed_canonical_c2ws  = contract_c2w_tran(canonical_c2ws,self.bbox)
            normed_canonical_c2ws = normed_canonical_c2ws.unsqueeze(0).repeat(self.num_points, 1)
            delta_xyz_ref, delta_quat_ref, delta_scale_ref = self.deform_network_ref(normed_canonical_means.data,normed_canonical_c2ws.data)
            delta_xyz += delta_xyz_ref
            delta_quat += delta_quat_ref
            delta_scale += delta_scale_ref
        else:
            ast_noise = torch.randn(1, 1, device=self.device).expand(self.num_points, -1) * self.time_interval * self.smooth_term(self.step)
            delta_xyz, delta_quat, delta_scale = self.deform_network(normed_canonical_means.data, t + ast_noise)
        
        return delta_xyz, delta_quat, delta_scale
    
    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        delta_xyz, delta_quat, delta_scale = None, None, None
        if self.defrom_gs:
            delta_xyz, delta_quat, delta_scale = self.get_deformation(self._means,cam.camtoworlds,cam.ref_flag)
            if self.delta_xyz_rescale:
                delta_xyz = delta_xyz * self.scene_scale
        
        if delta_xyz is not None:
            world_means = self._means + delta_xyz
        else:
            world_means = self._means
        
        if delta_quat is not None:
            world_quats = self.get_quats + delta_quat
        else:
            world_quats = self.get_quats
        
        if delta_scale is not None:
            activated_scales = torch.exp(self._scales + delta_scale)
        else:
            activated_scales = torch.exp(self._scales)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            if cam.ref_flag:
                n= self.sh_degree
            else:
                n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict

class DeformableGaussians_MIX_V4(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.deform_network = DeformNetwork(
            input_ch=3, **self.networks_cfg
        ).to(self.device)
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        self.delta_xyz_rescale = self.ctrl_cfg.get("delta_xyz_rescale", True)

        self.deform_network_ref = DeformNetwork_MIX_V4(
            input_ch=3, **self.networks_cfg
        ).to(self.device)
        # self.smooth_term_gen = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        # self.delta_xyz_rescale = self.ctrl_cfg.get("delta_xyz_rescale", True)
    
    @property
    def defrom_gs(self):
        if self.step < self.ctrl_cfg.coarse_train_interval:
            return False
        else:
            return True
    
    def set_bbox(self, bbox: torch.Tensor):
        self.bbox = bbox.reshape(2, 3)
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id
   
    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
        self.time_interval = 1 / len(normalized_timestamps)
        
    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        super().create_from_pcd(init_means, init_colors)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        # param_groups[self.class_prefix+"deform_network_ref"] = list(self.deform_network_ref.parameters())
        return param_groups
    
    def get_param_groups_ref(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        param_groups[self.class_prefix+"deform_network_ref"] = list(self.deform_network_ref.parameters())
        return param_groups
    
    @torch.no_grad()
    def get_deformation(self, canonical_means) -> Tuple:
        """
        get the deformation of the nonrigid instances
        """        
        t = self.normalized_timestamps[self.cur_frame]
        t = t.unsqueeze(0).repeat(self.num_points, 1)
        normed_canonical_means = contract(canonical_means, self.bbox) # range from 0 to 1
        ast_noise = torch.randn(1, 1, device=self.device).expand(self.num_points, -1) * self.time_interval * self.smooth_term(50000)
        
        delta_xyz, delta_quat, delta_scale = self.deform_network(normed_canonical_means.data, t + ast_noise)
        
        return delta_xyz, delta_quat, delta_scale
    
    def get_deformation_ref(self, canonical_means,canonical_c2ws) -> Tuple:
        """
        get the deformation of the nonrigid instances
        """        
        normed_canonical_means = contract(canonical_means, self.bbox) # range from 0 to 1  
        normed_canonical_c2ws  = contract_c2w(canonical_c2ws,self.bbox)
        normed_canonical_c2ws = normed_canonical_c2ws.unsqueeze(0).repeat(self.num_points, 1)
        delta_xyz_ref = self.deform_network_ref(normed_canonical_means.data, normed_canonical_c2ws.data)
        # delta_xyz += delta_xyz_ref
        # delta_quat += delta_quat_ref
        # delta_scale += delta_scale_ref
        
        
        return delta_xyz_ref#, delta_quat, delta_scale
   
    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        delta_xyz, delta_quat, delta_scale = None, None, None
        if self.defrom_gs:
            delta_xyz, delta_quat, delta_scale = self.get_deformation(self._means)
            if self.delta_xyz_rescale:
                delta_xyz = delta_xyz * self.scene_scale
        
        if delta_xyz is not None:
            world_means = self._means + delta_xyz
        else:
            world_means = self._means
        if cam.ref_flag:
            delta_xyz_ref = self.get_deformation_ref(world_means,cam.camtoworlds)
            world_means = world_means + delta_xyz_ref
        
        if delta_quat is not None:
            world_quats = self.get_quats + delta_quat
        else:
            world_quats = self.get_quats
        
        if delta_scale is not None:
            activated_scales = torch.exp(self._scales + delta_scale)
        else:
            activated_scales = torch.exp(self._scales)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict

class DeformableGaussians_MIX_V5(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.deform_network = DeformNetwork(
            input_ch=3, **self.networks_cfg
        ).to(self.device)
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
        self.delta_xyz_rescale = self.ctrl_cfg.get("delta_xyz_rescale", True)
    
    @property
    def defrom_gs(self):
        if self.step < self.ctrl_cfg.coarse_train_interval:
            return False
        else:
            return True
    
    def set_bbox(self, bbox: torch.Tensor):
        self.bbox = bbox.reshape(2, 3)
    
    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id
   
    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps
        self.time_interval = 1 / len(normalized_timestamps)
        
    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        super().create_from_pcd(init_means, init_colors)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix+"deform_network"] = list(self.deform_network.parameters())
        return param_groups
    
    def get_deformation(self, canonical_means) -> Tuple:
        """
        get the deformation of the nonrigid instances
        """        
        t = self.normalized_timestamps[self.cur_frame]
        t = t.unsqueeze(0).repeat(self.num_points, 1)
        normed_canonical_means = contract(canonical_means, self.bbox) # range from 0 to 1
        
        ast_noise = torch.randn(1, 1, device=self.device).expand(self.num_points, -1) * self.time_interval * self.smooth_term(self.step)
        
        delta_xyz, delta_quat, delta_scale = self.deform_network(normed_canonical_means.data, t + ast_noise)
        return delta_xyz, delta_quat, delta_scale
    
    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        delta_xyz, delta_quat, delta_scale = None, None, None
        if self.defrom_gs:
            delta_xyz, delta_quat, delta_scale = self.get_deformation(self._means)
            if self.delta_xyz_rescale:
                delta_xyz = delta_xyz * self.scene_scale

        if delta_xyz is not None:
            world_means = self._means + delta_xyz
        else:
            world_means = self._means
        
        if delta_quat is not None:
            world_quats = self.get_quats + delta_quat
        else:
            world_quats = self.get_quats
        
        if delta_scale is not None:
            activated_scales = torch.exp(self._scales + delta_scale)
        else:
            activated_scales = torch.exp(self._scales)
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_rotations = self.quat_act(world_quats)
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict
 