# Wildlife Species Classifier for Preservation Efforts

**Dataset:** [Animal Species Classification \- V3](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset) (Kaggle, utkarshsaxenadn)  
**Architecture:** MobileNetV2 (Transfer Learning, CNN)  
**Framework:** PyTorch

---

## Project Overview

Traditional wildlife monitoring requires researchers to manually review hours of camera-trap footage to identify which animal species appeared and when. This project trains a CNN to automatically classify animal species from images, enabling:

* Instant species labeling with little to no manual review  
* Searchable, timestamped logs of animal sightings  
* Scalable monitoring across many cameras simultaneously

### Why Deep Learning?

Animal species recognition is a **visually complex, high-variance problem** — the same species looks different across lighting conditions, angles, partial occlusion, and seasonal coat changes. Traditional approaches fall short:

| Approach | Weakness |
| ----- | ----- |
| Rule-based / template matching | Fails with pose/scale/lighting variation |
| Classical ML (SVM \+ HOG/SIFT) | Manual feature engineering, poor generalization |
| **CNNs (deep learning)** | **Automatically learns hierarchical visual features; handles natural variation** |

CNNs excel here because early layers learn edges and textures, middle layers learn shapes and patterns, and final layers learn species-discriminative features — all without manual design.

### Architecture: MobileNetV2 \+ Custom Classifier Head

We use **transfer learning**: a MobileNetV2 backbone pretrained on ImageNet (1.2M images, 1000 classes) provides powerful low-level feature extraction. We freeze the backbone and only train a new classifier head for our species labels.

The backbone is designed to be frozen and reused, so the model can be finetuned to a specific region or ecosystem by retraining only the classifier head on local species data.

<img width="680" height="990" alt="mobilenetv2_architecture_diagram" src="https://github.com/user-attachments/assets/8e62a79a-cd9a-48a9-bbb6-5d006ac3727e" />


**Why MobileNetV2?** It is efficient (designed for mobile/edge deployment), accurate, and fast to fine-tune — ideal for a student project that needs to run on a laptop CPU or a single GPU.

## Project Summary

### Dataset

**Animal Species Classification \- V3** by Utkarsh Saxena (Kaggle).  
Structure: one folder per species under Train/ and Test/; all images pre-resized to 224×224 RGB. We pool and re-split to 70/15/15.

### Training Method

Transfer learning from MobileNetV2 pretrained on ImageNet. The backbone is frozen; only the custom classifier head is trained. Optimiser: Adam (lr=1e-3, weight\_decay=1e-4). Loss: CrossEntropyLoss. Scheduler: ReduceLROnPlateau (halves LR after 3 stagnant epochs). Early stopping after 5 stagnant epochs.

### Evaluation

Reported on the held-out test set (never seen during training or hyperparameter tuning):

#### Accuracy
top-1 percentage correct.

  ##### Test Data Results:
  <img width="1291" height="736" alt="Screenshot 2026-05-11 134239" src="https://github.com/user-attachments/assets/01993f31-80c5-4249-abd7-d554ce7fa45e" />


  ##### Your-Image-Input results:
<img width="1277" height="727" alt="Screenshot 2026-05-11 134246" src="https://github.com/user-attachments/assets/6fe87c9b-55bc-4130-89dd-f29ca9db91ba" />


#### Precision / Recall / F1
per-class and macro-averaged.
<img width="481" height="432" alt="Screenshot 2026-05-11 133738" src="https://github.com/user-attachments/assets/033df543-ae35-47d1-83da-1ff2c51bca9d" />


#### Confusion matrix
row-normalised to show per-class recall.
<img width="1649" height="1483" alt="confusion_matrix" src="https://github.com/user-attachments/assets/bfed6483-8305-4d24-bf5a-6290a1993871" />



### Strengths of Deep Learning Here

* No manual feature engineering required.  
* Transfer learning allows strong performance even with a modest dataset.  
* Generalises across lighting, angle, and background variation.  
* The model cannot classify species outside its training set. For rare or newly discovered species with few available images, few-shot learning techniques (e.g. Prototypical Networks) would be needed.

### Weaknesses

* Requires labelled training data per species.  
* Performance degrades on species very different from those in training.  
* Cannot detect species not in the training classes.

### Future Work

* Location-specific finetuning: deploy the frozen backbone to a new region and retrain only the classifier head on local species data  
* Few-shot learning: extend the model to recognize new species from 1–5 examples  
* Real-time video integration: apply the classifier frame-by-frame on camera trap footage with timestamp logging  
* Human-in-the-loop integration: low-confidence predictions are flagged for expert review, enabling individual animal identification beyond species level and generating labelled data for continuous model improvement.

### Reference Articles

* Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR. [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)  
* Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML. [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)  
* Norouzzadeh et al. (2018). *Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning.* PNAS. [https://doi.org/10.1073/pnas.1719367115](https://doi.org/10.1073/pnas.1719367115)  
* He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR. [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)  
* Snell et al. (2017). Prototypical Networks for Few-shot Learning. NeurIPS. [https://arxiv.org/abs/1703.05175](https://arxiv.org/abs/1703.05175)

### GitHub

[https://github.com/Karczel/Wildlife-Species-Classifier-for-Preservation-Efforts](https://github.com/Karczel/Wildlife-Species-Classifier-for-Preservation-Efforts)

# Running the code

Use python 3.11 venv for PyTorch and Jupyter compatibility.

Download ipykernel when prompted (VS Code)

This model is trained on [Animal Species Classification - V3](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset), you may download the zip file separately, or use the Legacy Kaggle API call code blocks, by dropping your kaggle.json in the root folder.
