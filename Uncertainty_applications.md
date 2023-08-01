## NeRF uncertainty application

### Active Learning

Related papers: (In the order of time)
- [ICRA 2016] An information gain formulation for active volumetric 3D reconstruction. [Paper](https://rpg.ifi.uzh.ch/docs/ICRA16_Isler.pdf)    
    **TLDR**: consider the problem of *Next Best Views* (NBV) selection, volumetric reconstruction of an object, using the *Information Gain* (IG) metric, 
- [ECCV 2022] Active View Planning for Radiance Fields [Paper](https://imrss2022.github.io/contributions/lin.pdf)
- [Arxiv 2022.11] ActiveRMAP: Radiance Field for Active Mapping And Planning. [Paper](https://arxiv.org/pdf/2211.12656.pdf)
- [ICRA 2023] Density-aware NeRF Ensembles: Quantifying Predictive Uncertainty in Neural Radiance Fields. [Paper](https://arxiv.org/pdf/2209.08718.pdf) (see issues)
- [IROS 2023] NeU-NBV: Next Best View Planning Using Uncertainty Estimation
in Image-Based Neural Rendering. [Paper](https://arxiv.org/pdf/2303.01284.pdf) [Code](https://github.com/dmar-bonn/neu-nbv)
- [RAL 2023] NeurAR: Neural Uncertainty for Autonomous 3D
Reconstruction with Implicit Neural Representations. [Paper](https://arxiv.org/pdf/2207.10985.pdf)
- [RAL 2023] Uncertainty Guided Policy for Active Robotic 3D
Reconstruction using Neural Radiance Fields. [Paper](https://arxiv.org/pdf/2209.08409.pdf)
- [Arxiv 2023.7] Active Implicit Object Reconstruction using Uncertainty-guided Next-Best-View Optimziation. [Paper]()[Code]()

### NeRF + Robot Navigation 
- [RAL 2022] Vision-Only Robot Navigation in a Neural Radiance World. [Paper](https://arxiv.org/pdf/2110.00168.pdf)
  **TLDR**: Based on a offine pre-trained NeRF model, introduce a trajectory optimization algorithm that avoids collisions with high-density regions

Other fields:
- [SLAM] iMAP: Implicit Mapping and Positioning in Real-Time [[Project](https://edgarsucar.github.io/iMAP/)] (see issues)
