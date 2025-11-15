### MoCo-INR


This is the official code repository of our work "Unsupervised Motion-Compensated Decomposition for Cardiac MRI
Reconstruction via Neural Representation"

---

### Overview
![Pipeline of MoCo-INR](./Fig/MoCo-INR_Pipeline.png)
*Figure 1. Pipeline of MoCo-INR*.


### Results

![VISTA_recon](Fig/VISTA_recon.gif)
*Figure 2. Qualitative and quantitative comparison with MoCo-INR on the SAX view using the VISTA sampling pattern (AF=20)*.

![GA_recon](Fig/GA_recon.gif)
*Figure 3. Qualitative and quantitative comparison with MoCo-INR on the SAX view using the Golden-Angle Radial sampling (#spoke/frame=3)*.


![FreeBeathing](Fig/FreeBreathing_recon.gif)
*Figure 4. Qualitative comparison with MoCo-INR on free-breathing scans*.


### Run Recon Demos

We provide recon demo to demonstrate how MoCo-INR works. 
- `demo_VISTA_recon.ipynb` shows the reconstruction workflow of VISTA sampling pattern

Work in progress — updates coming soon…
- `demo_GA_recon.ipynb` shows the reconstruction workflow of Golden-Angle Radial sampling pattern
- `demo_FB_recon.ipynb` shows the reconstruction workflow of free-breathing scans

