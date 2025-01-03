
<div align="center">   
  
# DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation
</div>

 
## [Project Page](https://drivedreamer4d.github.io/) | [Paper]()

# News
- **[2024/10/17]** Repository Initialization.

# Abstract 

Closed-loop simulation is essential for advancing end-to-end autonomous driving systems. Contemporary sensor simulation methods, such as NeRF and 3DGS, rely predominantly on conditions closely aligned with training data distributions, which are largely confined to forward-driving scenarios. Consequently, these methods face limitations when rendering complex maneuvers (e.g., lane change, acceleration, deceleration). Recent advancements in autonomous-driving world models have demonstrated the potential to generate diverse driving videos. However, these approaches remain constrained to 2D video generation, inherently lacking the spatiotemporal coherence required to capture intricacies of dynamic driving environments.
In this paper, we introduce **DriveDreamer4D**, which enhances 4D driving scene representation leveraging world model priors. 
Specifically, we utilize the world model as a data machine to synthesize novel trajectory videos, where structured conditions are explicitly leveraged to control the spatial-temporal consistency of traffic elements. Besides, the cousin data training strategy is proposed to facilitate merging real and synthetic data for optimizing 4DGS. To our knowledge, **DriveDreamer4D** is the first to utilize video generation models for improving 4D reconstruction in driving scenarios.
Experimental results reveal that **DriveDreamer4D** significantly enhances generation quality under novel trajectory views, achieving a relative improvement in FID  by 32.1%, 46.4%, and 16.3% compared to PVG, S3Gaussian, and Deformable-GS. Moreover, **DriveDreamer4D** markedly enhances the spatiotemporal coherence of driving agents, which is verified by a comprehensive user study and the relative increases of 22.6%, 43.5%, and 15.6% in the NTA-IoU metric.

# DriveDreamer4D Framework

<img width="1349" alt="method" src="https://github.com/user-attachments/assets/a09f5e09-450b-4d39-8388-c58b2925fece">

# Install
```
conda create -n drivedreamer4d python=3.8
conda activate drivedreamer4d
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install the glm library. This provides the header files needed to compile gsplat.
sudo apt-get install libglm-dev

# Verify glm installation  
ls /usr/include/glm  

# RTX4090:8.9, A100:8.0
export TORCH_CUDA_ARCH_LIST="8.0 8.9"

# Confirm 
echo $TORCH_CUDA_ARCH_LIST  

# Check compiler compatibility
nvcc --version
g++ --version 

# For CUDA 12.1, it is safe to use gcc/g++ 11 or lower.
# If gcc is the latest version, install and replace it with the appropriate version.
# Install gcc-11:
sudo apt-get install gcc-11 g++-11

# Change default gcc/g++ settings
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
sudo update-alternatives --config gcc
sudo update-alternatives --config g++

# check g++ version
g++ --version

pip install ./submodules/gsplat-1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install ./submodules/nvdiffrast
pip install ./submodules/smplx
```
# Prepare
Download data ([Baidu](https://pan.baidu.com/s/18huHwBVOu0T796NXLt1LCA?pwd=5zpc), [Google](https://drive.google.com/drive/folders/1gVrs4FbMUwb40L-4dPvM7V8nzKBWza2Z?usp=sharing))  and extract it to the ./data/waymo/ directory.
```shell
cd data
ln -s /home/hyunkoo/Dataset/NAS/nfsRoot/Datasets/Waymo_Datasets/DriveDreamer4D/data/waymo waymo 
```

Download checkpoint ([Baidu](https://pan.baidu.com/s/18huHwBVOu0T796NXLt1LCA?pwd=5zpc), [Google](https://drive.google.com/drive/folders/1gVrs4FbMUwb40L-4dPvM7V8nzKBWza2Z?usp=sharing)) to ./exp/pvg_example
```shell
cd exp/pvg_example/
ln -s /home/hyunkoo/Dataset/NAS/nfsRoot/Datasets/Waymo_Datasets/DriveDreamer4D/exp/pvg_example/checkpoint_final.pth checkpoint_final.pth
```

# Render

#### for this process, I updated a code at line 889, `DriveDreamer4D/models/trainers/base.py` because of following error message:
```shell
state_dict = torch.load(ckpt_path)
Traceback (most recent call last):
  File "tools/eval.py", line 283, in <module>
    main(args)
  File "tools/eval.py", line 213, in main
    trainer.resume_from_checkpoint(
  File "/home/hyunkoo/DATA/HDD8TB/Add_Objects_DrivingScense/DriveDreamer4D/models/trainers/base.py", line 903, in resume_from_checkpoint
    self.load_state_dict(state_dict, load_only_model=load_only_model, strict=strict,step=step)
  File "/home/hyunkoo/DATA/HDD8TB/Add_Objects_DrivingScense/DriveDreamer4D/models/trainers/base.py", line 888, in load_state_dict
    msg = super().load_state_dict(state_dict, strict)
  File "/home/hyunkoo/anaconda3/envs/drivedreamer4d/lib/python3.8/site-packages/torch/nn/modules/module.py", line 2215, in load_state_dict
    raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
RuntimeError: Error(s) in loading state_dict for SingleTrainer:
        Missing key(s) in state_dict: "lpips.net.scaling_layer.shift", "lpips.net.scaling_layer.scale". 
```
```python
# msg = super().load_state_dict(state_dict, strict)
msg = super().load_state_dict(state_dict, strict=False) # hkkim
```

```
python tools/eval.py --resume_from ./exp/pvg_example/checkpoint_final.pth
```

Create demo
```shell
cd utils
python create_demo_video.py
```

<img width="1349" alt="method" src="docs/figures/pvg_example.gif">


# Scenario Selection

All selected scenes are sourced from the validation set of the Waymo dataset. The official file names of these scenes, are listed along with their respective starting and ending frames.
| Scene | Start Frame | End Frame |
| :-----| :----: | :----: |
| segment-10359308928573410754_720_000_740_000_with_camera_labels.tfrecord | 120 | 159 |
| segment-12820461091157089924_5202_916_5222_916_with_camera_labels.tfrecord | 0 | 39 |
|segment-15021599536622641101_556_150_576_150_with_camera_labels.tfrecord|0|39|
|segment-16767575238225610271_5185_000_5205_000_with_camera_labels.tfrecord|0|39|
|segment-17152649515605309595_3440_000_3460_000_with_camera_labels.tfrecord|60|99|
|segment-17860546506509760757_6040_000_6060_000_with_camera_labels.tfrecord|90|129|
|segment-2506799708748258165_6455_000_6475_000_with_camera_labels.tfrecord|80|119|
|segment-3015436519694987712_1300_000_1320_000_with_camera_labels.tfrecord|40|79|


# Rendering Results in Lane Change Novel Trajectory

<div align="center">   
  
https://github.com/user-attachments/assets/2431a910-80be-4548-899c-37acd8bded8d

</div>

<div align="center">   

https://github.com/user-attachments/assets/710eca85-5d98-4155-b07e-b491f23239ed

</div>

<div align="center">   
  
https://github.com/user-attachments/assets/d6d7d3cd-40d0-44ee-be05-8d2e1f183281

</div>

**Comparisons of novel trajectory renderings during lane change scenarios. The left column shows <b>PVG</b>, <b><span>S<sup>3</sup>Gaussian</span></b>, and <b>Deformable-GS</b>, while the right column shows <b><em>DriveDreamer4D</em>-PVG</b>, <b><em>DriveDreamer4D</em>-<span>S<sup>3</sup>Gaussian</span></b>, and <b><em>DriveDreamer4D</em>-Deformable-GS</b>.**

# Rendering Results in Speed Change Novel Trajectory
<div align="center">   

https://github.com/user-attachments/assets/f75e0f87-7d51-48db-9750-27fa53366f49

</div>

<div align="center">   

https://github.com/user-attachments/assets/1b18fcfa-a869-41b9-9a74-622d6cb84212

</div>

<div align="center">   

https://github.com/user-attachments/assets/eab7b98c-0466-4459-a927-d314d2ece1ce

</div>

**Comparisons of novel trajectory renderings during speed change scenarios. The left column shows <b>PVG</b>, <b><span>S<sup>3</sup>Gaussian</span></b>, and <b>Deformable-GS</b>, while the right column shows <b><em>DriveDreamer4D</em>-PVG</b>, <b><em>DriveDreamer4D</em>-<span>S<sup>3</sup>Gaussian</span></b>, and <b><em>DriveDreamer4D</em>-Deformable-GS</b>.**

# Acknowledgements
We would like to thank the following works and projects, for their open research and exploration: [DriveStudio](https://github.com/ziyc/drivestudio), [DriveDreamer](https://github.com/JeffWang987/DriveDreamer), 
[DriveDreamer-2](https://github.com/f1yfisher/DriveDreamer2), and [DriveDreamer4D](https://github.com/GigaAI-research/DriveDreamer4D).

# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@inproceedings{zhao2024drive,
    title={DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation}, 
    author={Guosheng Zhao and Chaojun Ni and Xiaofeng Wang and Zheng Zhu and Xueyang Zhang and Yida Wang and Guan Huang and Xinze Chen and Boyuan Wang and Youyi Zhang and Wenjun Mei and Xingang Wang},
    journal={arxiv arXiv preprint arXiv:2410.13571},
    year={2024},
}


