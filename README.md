# RAMR: Residual-Aware Meta-learning Re-weighting Framework
﻿
## 📊 Data Availability & Reproducibility
To strictly ensure the reproducibility of our experiments while adhering to the data distribution policies of the primary benchmarks, we provide the **fully processed subsets** of the Waymo dataset utilized in this study. 
﻿
The provided `.npz` files contain the extracted kinematic features, baseline physical prior predictions (e.g., Gipps model), and the historical residual sequences. **These trajectories were rigorously extracted and processed from the public FollowNet benchmark (Chen et al., 2023)**, strictly following the trajectory filtering criteria, parameter calibration (SLSQP), and feature extraction pipeline detailed in **Section 4.1-4.3** of our manuscript.
﻿
**Included Data Files:**
* `waymo_ego_acceleration_ego_accelerations.npz`: Ground-truth human driving accelerations.
* `Waymo_Gipps_analysis_final_extracted_features.npz`: Instantaneous physical state features (spacing, ego speed, relative speed).
* `Waymo_Gipps_analysis_final_residuals.npz`: Historical error residuals between the Gipps physical prior and human ground truth.
