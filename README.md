
<div align="center">   
  
# DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation
</div>

 
## [Project Page](https://drivedreamer4d.github.io/) | [Paper]()

# News
- **[2024/10/17]** Repository Initialization.

# Abstract 

Closed-loop simulation is essential for advancing end-to-end autonomous driving systems. Contemporary sensor simulation methods, such as NeRF and 3DGS, rely predominantly on conditions closely aligned with training data distributions, which are largely confined to forward-driving scenarios. Consequently, these methods face limitations when rendering complex maneuvers (e.g., lane change, acceleration, deceleration).Recent advancements in autonomous-driving world models have demonstrated the potential to generate diverse driving videos. However, these approaches remain constrained to 2D video generation, inherently lacking the spatiotemporal coherence required to capture intricacies of dynamic driving environments. In this paper, we introduce **DriveDreamer4D**, which enhances 4D driving scene representation leveraging world model priors. Specifically, we utilize the world model as a data machine to synthesize novel trajectory videos based on real-world driving data. Notably, we explicitly leverage structured conditions to control the spatial-temporal consistency of foreground and background elements, thus the generated data adheres closely to traffic constraints. To our knowledge, **DriveDreamer4D** is the first to utilize video generation models for improving 4D reconstruction in driving scenarios. Experimental results reveal that **DriveDreamer4D** significantly enhances generation quality under novel trajectory views, achieving a relative improvement in FID by 24.5%, 39.0%, and 10.5% compared to PVG, S3Gaussian, and Deformable-GS. Moreover, **DriveDreamer4D** markedly enhances the spatiotemporal coherence of driving agents, which is verified by a comprehensive user study and the relative increases of 19.7%, 12.7%, and 11.3% in the NTA-IoU metric.

# DriveDreamer4D Framework

<img width="1349" alt="method" src="https://github.com/user-attachments/assets/a09f5e09-450b-4d39-8388-c58b2925fece">

# Scenario Selection

The eight scenarios selected are as follows: 005, 018, 027, 065, 081, 096, 121 and 164 in the validation set of Waymo.


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

# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@inproceedings{zhao2024drive,
    title={DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation}, 
    author={Guosheng Zhao and Chaojun Ni and Xiaofeng Wang and Zheng Zhu and Xueyang Zhang and Yida Wang and Guan Huang and Xinze Chen and Boyuan Wang and Youyi Zhang and Wenjun Mei and Xingang Wang},
    journal={arxiv arXiv preprint arXiv:2410.13571},
    year={2024},
}


