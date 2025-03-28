# PerspectAI Face Verification System

This repository contains implementations of a lightweight face verification system designed for PerspectAI, achieving significantly better accuracy than the previous deployed model (which had 75-85% accuracy).

## Overview

The main implementation file is **PerspectAI.Face_Verification-Copy1.ipynb**, which implements a complete face verification pipeline. Additionally, the repository contains supporting scripts:

1. **FaceVerify_2DPCA_MLP.ipynb** (formerly Script_128-2.ipynb): A high-accuracy model trained on a dataset that generates 21,248 training pairs (10,624 positive pairs and 10,624 negative pairs).
2. **FaceVerify_Enhanced_Ensemble.ipynb** (formerly Script_final_gen.ipynb): A more robust model trained on a larger dataset with advanced ensemble techniques.

All implementations follow similar workflows but differ in dataset size, feature extraction techniques, and classification methods.

## System Workflow

### 1. Face Detection & Processing
- Utilizes Viola-Jones face detection (OpenCV's Haar Cascade) to identify the Region of Interest (ROI)
- Upscales the detected face images using ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)
- Converts detected faces to grayscale
- Resizes images to 64x64 pixels (with potential improvements at 128x128 or 256x256)
- Organizes processed images in structured directories

### 2. Feature Extraction with 2DPCA
- Computes covariance matrix from training images
- Extracts eigenvectors (eigenfaces) using 2DPCA, with k=30 providing optimal results
- Projects face images onto lower-dimensional eigenspace
- Stores eigenvectors in pickle files (e.g., 'eig_vecsk_128.pkl', 'eig_vecsk_128sept10.pkl')

### 3. Pair Generation
- Creates positive pairs (same identity) and negative pairs (different identities)
- Projects all faces onto the PCA eigenspace
- Ensures balanced datasets with equal numbers of positive and negative pairs
- Training set: 21,248 pairs (10,624 positive + 10,624 negative)
- Validation set: 5,160 pairs (2,580 positive + 2,580 negative)
- Test set: 8,112 pairs (4,056 positive + 4,056 negative)

### 4. Classification
Multiple classifiers were evaluated for optimal performance:

#### FaceVerify_2DPCA_MLP.ipynb
- **MLP Classifier** (best performance): 
  - Architecture: hidden_layer_sizes=(256,128,64)
  - Validation accuracy: 87.03%, Validation precision: 91.23%
  - Test accuracy: 93.23%, Test precision: 91.32%
  - For class 0: Precision 0.95, Recall 0.92, F1-score 0.93
  - For class 1: Precision 0.92, Recall 0.96, F1-score 0.94
  - Saves the final model as 'mlp_classifier.pkl'

- **SVM** with RBF kernel: 
  - Implements grid search with cross-validation
  - Optimal parameters: C=10, gamma=1e-08
  - Saves as 'svc_rbf_c_10_gamma_1e-08.pkl'

#### FaceVerify_Enhanced_Ensemble.ipynb
- **Random Forest Classifier**:
  - Validation accuracy: 83.08%, Validation precision: 93.56%
  - Test accuracy: 93.09%, Test precision: 93.56%
  - Saves as 'rf_classifier10000.pkl'

- **XGBoost Classifier**:
  - objective='binary:logistic', n_estimators=8000
  - Validation accuracy: 88.64%, Validation precision: 92.06%
  - Test accuracy: 92.77%, Test precision: 90.61%

- **KNN Classifier** (k=20):
  - Validation accuracy: 81.98%, Validation precision: 77.08%
  - Test accuracy: 82.92%, Test precision: 77.08%

## Computational Efficiency

The MLP classifier demonstrates exceptional efficiency:
- **Inference time**: ~0.038 seconds average
- **Memory usage**: ~23.77 MB peak
- **CPU utilization**: ~19.79% average
- **AUC-ROC**: 0.93540
- **Equal Error Rate (EER)**: 4.438%

The confusion matrix indicates:
- True Positives: 3,876
- True Negatives: 3,712
- False Positives: 344
- False Negatives: 180

## Deployment Capability

A key feature of this implementation is its ability to be deployed locally on a candidate's system instead of requiring server-side processing. This aligns with PerspectAI's goal of creating lightweight verification systems that can run efficiently on client machines, ensuring:

- Enhanced privacy (face data remains on the local machine)
- Reduced server infrastructure requirements
- Real-time verification capability (low latency)
- Lower resource usage compared to server-dependent solutions

## Requirements

- Python 3.x
- OpenCV (for Viola-Jones face detection)
- NumPy
- scikit-learn
- tqdm (for progress bars)
- matplotlib (for visualization)
- XGBoost (for FaceVerify_Enhanced_Ensemble.ipynb)
- PyTorch (for ESRGAN upscaling)

## Dataset Structure

The system expects data organized in directories:
```
dataset/
  ├── train_ds/
  │   ├── person1/
  │   │   ├── image1.jpg
  │   │   └── ...
  │   └── person2/
  │       └── ...
  ├── val_ds/
  │   └── ...
  └── test_ds/
      └── ...
```

## Model Files

The system generates several key files:

- **eig_vecsk_128.pkl**: PCA eigenvectors for face representation (k=30)
- **eig_vecsk_128sept10.pkl**: Larger set of PCA eigenvectors for enhanced model
- **mlp_classifier.pkl**: MLP model (best performer with 93.23% accuracy)
- **svc_rbf_c_10_gamma_1e-08.pkl**: SVM model 
- **rf_classifier10000.pkl**: Random Forest model

## Common Failure Cases & Solutions

Most false negatives were due to:
- Wrongly detected ROIs by the Viola-Jones face detection algorithm
- Partial faces in webcam view
- Complex backgrounds with patterns that confuse the face detector

Recommendations for improved accuracy:
- Have candidates sit in front of solid-color backgrounds
- Ensure full face is visible to the camera
- Consider alternative face detection methods like YOLO or MTCNN

## Promising Areas for Further Development

1. **Resolution Enhancement**: Resize images to 128x128 or 256x256 pixels instead of 64x64 to better leverage ESRGAN upscaling benefits (current implementation shows 1% accuracy increase with upscaling)

2. **Dataset Expansion**: Train on more diverse images to improve principle component representation and model generalization

3. **Face Detection Improvement**: Experiment with alternative face detection models like YOLO or MTCNN

4. **Demographic Considerations**: 
   - Ensure principle components represent diverse populations
   - Consider training region-specific models for areas with distinct facial characteristics

5. **Environment Optimization**: Guide candidates to use solid-color backgrounds to improve face detection accuracy

## References

1. Yang, J., Zhang, D., Frangi, A. F., & Yang, J. Y. (2004). Two-dimensional PCA: a new approach to appearance-based face representation and recognition. IEEE transactions on pattern analysis and machine intelligence, 26(1), 131-137.
2. Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. In Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition.
3. Wang, X. et al. (2019). ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. In: Computer Vision – ECCV 2018 Workshops.

For more detailed information, please refer to the Project Report PDF included in this repository. 