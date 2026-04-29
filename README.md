# AFM: Adversarial Flow Matching for Imperceptible Attacks on End-to-End Autonomous Driving

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/xxxx.xxxxx) 
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**AFM** is a novel **gray‑box adversarial attack** against end‑to‑end autonomous driving agents.  
It exploits the structural vulnerabilities of **Transformer backbones** (used by both monolithic VLA models and modular architectures) and generates **visually imperceptible** adversarial examples in **one step** (1‑NFE) via a neural average velocity field.

> 🔥 **Key highlights**  
>
> - First application of **Flow Matching** for adversarial attacks on autonomous driving.  
> - **Gray‑box** setting: only requires knowing that the victim uses a Transformer module (no full weights or gradients).  
> - **One‑step generation** – 40% faster than diffusion‑based attacks, with state‑of‑the‑art imperceptibility.  
> - Effective on **VLA** (SimLingo) and **modular** (TransFuser) agents in both open‑loop and closed‑loop (CARLA/Bench2Drive) evaluations.

---

## ✨ Attack Overview

<table>
<tr>
<td width="60%">
The proposed <strong>Adversarial Flow Matching (AFM)</strong> framework:
<ul>
  <li>Inverts a clean image into the latent space via a frozen VAE encoder.</li>
  <li>Injects <strong>dual perturbations</strong>: <code>δ<sub>z</sub></code> (latent space) and <code>δ<sub>u</sub></code> (neural velocity field).</li>
  <li>Performs <strong>single‑step ODE update</strong> (1‑NFE) using a pre‑trained Flow Matching network.</li>
  <li>Optimizes an <strong>attention‑guided loss</strong> that focuses on road regions and high‑saliency tokens.</li>
</ul>
The resulting adversarial image remains virtually indistinguishable from the clean input but forces the target AD agent into hazardous maneuvers (e.g., off‑road, collisions).
</td>
<td width="40%" align="center">
  < img src="AFM_framework.png" alt="AFM pipeline" width="100%">
  <br>
  <em>Fig. 2 from the paper – AFM attack mechanism</em>
</td>
</tr>
</table>


---
![AFM pipeline](main/AFM.png)
*Fig. 2 from the paper – AFM attack mechanism*


## 📊 Experimental Results

### 🧪 Experimental Setup

#### 🤖 Representative Models

We evaluate AFM on two distinct end‑to‑end AD paradigms:

- **TransFuser** – a *specialized modular architecture* that fuses RGB and LiDAR BEV using Transformer modules at intermediate layers.  
- **SimLingo** – a *monolithic VLA model* built on InternVL2 + Qwen2 LLM, where we attack the Vision Transformer (ViT) component.

Both rely on Transformer backbones – the only prior knowledge required by our gray‑box attack.

#### 📁 Datasets & Scenarios

| Model      | Dataset                  | Frames     | Scenarios                                        |
| ---------- | ------------------------ | ---------- | ------------------------------------------------ |
| TransFuser | CARLA IL (Chitta et al.) | 228k @ 2Hz | Complex (dense intersections) / Common (highway) |
| SimLingo   | PDM‑lite (Renz et al.)   | 3.1M @ 4Hz | Daytime / Nighttime                              |

Closed‑loop evaluation uses **Bench2Drive** (10 routes, CARLA).

#### ⚙️ Implementation Details

- Perturbation bounds: `ε_z = 0.03`, `ε_u = 0.03`
- Loss weights: `λ_f = 3.0` (road‑focused), `λ_a = 4.5` (attention), `λ_c = 6.0` (latent constraint)
- Optimizer: Adam, 50 iterations, `η_z = η_u = 0.05`
- Hardware: RTX 4090 (open‑loop), A800 80G (closed‑loop)

---

### 📈 Quantitative Comparison

**Table I – Performance on TransFuser (modular)**

| Method         | SHIFT (m) ↑ | SR (%) ↑  | LPIPS ↓   | SSIM ↑    | FID ↓     | TIME (s) ↓ |
| -------------- | ----------- | --------- | --------- | --------- | --------- | ---------- |
| FGSM           | 4.89        | 89.71     | 0.370     | 0.708     | 64.19     | **0.14**   |
| PGD            | 5.99        | 92.03     | 0.359     | 0.732     | 58.43     | 1.23       |
| DiffAttack     | 2.47        | 60.12     | 0.165     | 0.871     | 36.57     | 10.61      |
| PerC‑AL        | 6.81        | **97.06** | 0.694     | 0.196     | 223.98    | 11.15      |
| NCF            | 0.52        | 15.34     | 0.256     | 0.879     | 34.22     | 3.07       |
| **AFM (ours)** | 4.93        | 88.24     | **0.141** | **0.881** | **23.18** | 6.75       |

> ✅ AFM achieves the **best visual imperceptibility** (lowest LPIPS, FID, highest SSIM) while maintaining high attack success (SR ≈ 88%). It is **40% faster** than DiffAttack.

**Table II – Performance on SimLingo (VLA)**

| Method         | SHIFT (m) ↑ | SR (%) ↑  | LPIPS ↓   | SSIM ↑    | FID ↓    | TIME (s) ↓ |
| -------------- | ----------- | --------- | --------- | --------- | -------- | ---------- |
| FGSM           | 2.53        | 78.18     | 0.395     | 0.734     | 54.01    | **0.48**   |
| PGD            | 6.51        | 96.26     | 0.326     | 0.798     | 45.67    | 1.80       |
| DiffAttack     | 1.66        | 59.92     | 0.114     | 0.928     | 20.61    | 9.98       |
| PerC‑AL        | 2.29        | 72.26     | 0.099     | 0.960     | 13.71    | 10.83      |
| NCF            | 0.91        | 28.48     | 0.237     | 0.848     | 33.09    | 4.56       |
| **AFM (ours)** | 3.20        | **87.14** | **0.075** | **0.959** | **8.10** | 6.83       |

> 🔥 On the more challenging VLA agent, AFM again **outperforms all baselines in imperceptibility** (LPIPS 0.075, FID 8.10) while achieving the **highest attack success rate** (87.14%).

---

### 🔄 Cross‑Model Transferability (Gray‑Box Setting)

We attack **without target gradients** – only knowledge that the victim uses a Transformer.

**Table III – Transfer between SimLingo (SL) and TransFuser (TF)**

| Direction | Method     | SHIFT (m) ↑ | SR (%) ↑  | LPIPS ↓   | SSIM ↑    |
| --------- | ---------- | ----------- | --------- | --------- | --------- |
| SL → TF   | PGD        | 0.316       | 8.40      | 0.476     | 0.763     |
|           | DiffAttack | 0.474       | 12.61     | 0.114     | 0.980     |
|           | **AFM**    | **0.506**   | **12.82** | **0.022** | **0.995** |
| TF → SL   | PGD        | 1.337       | 50.71     | 0.291     | 0.812     |
|           | **AFM**    | 1.192       | 46.68     | **0.148** | **0.871** |

- AFM achieves **state‑of‑the‑art transfer imperceptibility** (LPIPS as low as 0.022) while keeping attack success competitive.
- This verifies that **any Transformer‑based AD agent** is vulnerable, regardless of architecture (modular or VLA).

---

### 🚦 Closed‑Loop Evaluation (Bench2Drive – SimLingo)

Intermittent attack (every 10 frames) – temporal compounding of errors.

**Table IV – Closed‑loop performance and failure modes**

| Method     | RC (%) ↓ | Off‑Road ↑ | Collisions ↑ | Route Dev ↑ | Blocked ↑ | LPIPS ↓   | SSIM ↑    |
| ---------- | -------- | ---------- | ------------ | ----------- | --------- | --------- | --------- |
| Clean      | 100.0    | 0.0        | 0.0          | 0           | 0         | —         | —         |
| FGSM       | 42.33    | 14.97      | 2.71         | 0           | 0         | 0.594     | 0.511     |
| PGD        | 23.12    | 16.26      | 2.42         | 0           | 0         | 0.525     | 0.555     |
| DiffAttack | 5.14     | 18.63      | 0.60         | 0           | 70        | 0.117     | 0.915     |
| PerC‑AL    | 10.63    | 26.08      | 1.42         | 0           | 50        | 0.237     | 0.922     |
| NCF        | 0.00     | 0.00       | 0.00         | 0           | **100**   | 0.184     | 0.912     |
| **AFM**    | 17.29    | **40.06**  | **1.52**     | **20**      | 20        | **0.075** | **0.956** |

- AFM induces **active hijacking** (high Off‑Road, Route Deviation) – not just “static freezing” (Blocked rate only 20%).
- Maintains **unmatched stealth** under real driving dynamics.

---

### 🧩 Ablation Study

**Effect of perturbation budget `ε` (ε_z = ε_u)**

| ε    | SR (%) ↑ | SSIM ↑    | LPIPS ↓   |
| ---- | -------- | --------- | --------- |
| 0.01 | 68.5     | 0.972     | 0.042     |
| 0.03 | **87.1** | **0.959** | **0.075** |
| 0.05 | 91.2     | 0.921     | 0.132     |

- Larger `ε` improves attack success but degrades visual quality.  
- Our default `ε = 0.03` balances both objectives optimally.

---

## 📝 Conclusion

We presented **AFM**, the first **Flow Matching‑based adversarial attack** against end‑to‑end autonomous driving agents. Key outcomes:

- ✅ **Gray‑box** – only requires Transformer architecture prior, no full model access.  
- ✅ **One‑step generation** (1‑NFE) – 40% faster than diffusion attacks.  
- ✅ **State‑of‑the‑art imperceptibility** – lowest LPIPS/FID across both modular and VLA agents.  
- ✅ **Effective cross‑model transfer** – attacks transfer to unseen Transformer‑based models.  
- ✅ **Real‑world threat** – closed‑loop results show active hijacking (off‑road, route deviation) rather than trivial freezing.

### Limitations & Future Work

- **Performance gap** – white‑box attacks (e.g., PerC‑AL) can achieve slightly higher SR, but with much worse imperceptibility.  
- **Digital only** – physical world deployment (patches, V2X injection) is left for future work.  
- Next steps: bridge the gap to white‑box potency, and validate on physical vehicles.

---
