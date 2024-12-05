# CycleGAN

This repository contains an implementation of **CycleGAN**, a model architecture designed for unpaired image-to-image translation. The implementation strictly follows the architecture described in the official CycleGAN paper:  
**"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"** by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros.

## Introduction
CycleGAN enables image-to-image translation tasks without requiring paired examples. Unlike traditional GANs, it uses a cycle-consistency loss to ensure the mappings are consistent between the source and target domains.

---

## Features
- **Unpaired image-to-image translation**: Translate between two domains (e.g., summer to winter, horses to zebras) without paired datasets.
- **Cycle-consistency loss**: Ensures the mappings preserve essential image features.
- **Customizable architecture**: Modify generator, discriminator, or hyperparameters.

---

## Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/username/cycleGAN.git
   cd cycleGAN
   pip install -r requirements
   

---


## Prepare your data

Move your dataset into the /data directory.


---


Created by Siddharth Karmokar
