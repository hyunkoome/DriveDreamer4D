
<div align="center">   
  
# DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation
</div>

 
## [Project Page](https://drivedreamer4d.github.io/) | [Paper]()

# Abstract 

Closed-loop simulation is essential for advancing end-to-end autonomous driving systems. Contemporary sensor simulation methods, such as NeRF and 3DGS, rely predominantly on conditions closely aligned with training data distributions, which are largely confined to forward-driving scenarios. Consequently, these methods face limitations when rendering complex maneuvers (e.g., lane change, acceleration, deceleration).Recent advancements in autonomous-driving world models have demonstrated the potential to generate diverse driving videos. However, these approaches remain constrained to 2D video generation, inherently lacking the spatiotemporal coherence required to capture intricacies of dynamic driving environments. In this paper, we introduce **DriveDreamer4D**, which enhances 4D driving scene representation leveraging world model priors. Specifically, we utilize the world model as a data machine to synthesize novel trajectory videos based on real-world driving data. Notably, we explicitly leverage structured conditions to control the spatial-temporal consistency of foreground and background elements, thus the generated data adheres closely to traffic constraints. To our knowledge, **DriveDreamer4D** is the first to utilize video generation models for improving 4D reconstruction in driving scenarios. Experimental results reveal that **DriveDreamer4D** significantly enhances generation quality under novel trajectory views, achieving a relative improvement in FID by 24.5%, 39.0%, and 10.5% compared to PVG, S3Gaussian, and Deformable-GS. Moreover, **DriveDreamer4D** markedly enhances the spatiotemporal coherence of driving agents, which is verified by a comprehensive user study and the relative increases of 19.7%, 12.7%, and 11.3% in the NTA-IoU metric.


# News
- **[2024/10/17]** Repository Initialization.


# Rendering results in lane change novel trajectory


# Rendering results in speed change 




<!-- **WorldDreamer Framework**
<img width="1349" alt="method" src="https://github.com/JeffWang987/WorldDreamer/assets/49095445/0f95bde3-e19a-4b79-9bad-ee22e2cddeb1">
 -->


<!-- # Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{wang2023worlddreamer,
      title={WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens}, 
      author={Xiaofeng Wang and Zheng Zhu and Guan Huang and Boyuan Wang and Xinze Chen and Jiwen Lu},
      journal={arXiv preprint arXiv:2401.09985},
      year={2024}
} -->


