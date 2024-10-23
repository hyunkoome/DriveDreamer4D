
<div align="center">   
  
# DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation
</div>

 
## [Project Page](https://drivedreamer4d.github.io/) | [Paper]()

# News
- **[2024/10/17]** Repository Initialization.

# Abstract 

Closed-loop simulation is essential for advancing end-to-end autonomous driving systems. Contemporary sensor simulation methods, such as NeRF and 3DGS, rely predominantly on conditions closely aligned with training data distributions, which are largely confined to forward-driving scenarios. Consequently, these methods face limitations when rendering complex maneuvers (e.g., lane change, acceleration, deceleration).Recent advancements in autonomous-driving world models have demonstrated the potential to generate diverse driving videos. However, these approaches remain constrained to 2D video generation, inherently lacking the spatiotemporal coherence required to capture intricacies of dynamic driving environments. In this paper, we introduce **DriveDreamer4D**, which enhances 4D driving scene representation leveraging world model priors. Specifically, we utilize the world model as a data machine to synthesize novel trajectory videos based on real-world driving data. Notably, we explicitly leverage structured conditions to control the spatial-temporal consistency of foreground and background elements, thus the generated data adheres closely to traffic constraints. To our knowledge, **DriveDreamer4D** is the first to utilize video generation models for improving 4D reconstruction in driving scenarios. Experimental results reveal that **DriveDreamer4D** significantly enhances generation quality under novel trajectory views, achieving a relative improvement in FID by 24.5%, 39.0%, and 10.5% compared to PVG, S3Gaussian, and Deformable-GS. Moreover, **DriveDreamer4D** markedly enhances the spatiotemporal coherence of driving agents, which is verified by a comprehensive user study and the relative increases of 19.7%, 12.7%, and 11.3% in the NTA-IoU metric.

# DriveDreamer4D Framework
<img width="1349" alt="method" src="https://github.com/user-attachments/assets/c783ef23-45e8-4291-81a5-befceff2f539">

# Dataset Selection

The eight scenarios selected are as follows: 005, 018, 027, 065, 081, 096, 121, 164 in the validation set of Waymo.


# Rendering results in lane change novel trajectory

<div align="center">   
  
https://github.com/user-attachments/assets/3b328ada-c94e-46c1-9143-f26ef87394d1

</div>

<div align="center">   
  
https://github.com/user-attachments/assets/2921a374-8277-4176-9a45-49f8fd5f1ac1

</div>

<div align="center">   
  
https://github.com/user-attachments/assets/8ac2b579-d5f5-4315-9aeb-658d82915eb5

</div>

**Comparisons of novel trajectory renderings during lane change scenarios. The left column shows <b>PVG</b>, <b><span>S<sup>3</sup>Gaussian</span></b>, and <b>Deformable-GS</b>, while the right column shows <b><em>DriveDreamer4D</em>-PVG</b>, <b><em>DriveDreamer4D</em>-<span>S<sup>3</sup>Gaussian</span></b>, and <b><em>DriveDreamer4D</em>-Deformable-GS</b>.**

# Rendering results in speed change 
<div align="center">   
  
https://github.com/user-attachments/assets/b74ac2e4-6d2f-4af3-8ddc-2f484f912f20

</div>

<div align="center">   
  
https://github.com/user-attachments/assets/75448b73-85d5-4f35-8009-7487fc9c15ce

</div>

<div align="center">   
  
https://github.com/user-attachments/assets/f2e83cf8-3021-40ee-beab-6cab50aca8a8

</div>

**Comparisons of novel trajectory renderings during speed change scenarios. The left column shows <b>PVG</b>, <b><span>S<sup>3</sup>Gaussian</span></b>, and <b>Deformable-GS</b>, while the right column shows <b><em>DriveDreamer4D</em>-PVG</b>, <b><em>DriveDreamer4D</em>-<span>S<sup>3</sup>Gaussian</span></b>, and <b><em>DriveDreamer4D</em>-Deformable-GS</b>.**

# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@inproceedings{zhao2024drive,
    title={DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation}, 
    author={Guosheng Zhao and Chaojun Ni and Xiaofeng Wang and Zheng Zhu and Guan Huang and Xinze Chen and Boyuan Wang and Youyi Zhang and Wenjun Mei and Xingang Wang},
    journal={arxiv arXiv preprint arXiv:2410.13571},
    year={2024},
}


