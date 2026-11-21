# Pneumonia-Detection-Streamlit
Chest X-ray pneumonia classification using ResNet18, DenseNet121, and Streamlit
# AI-Pneumonia-Classification

## Deep Learning-Based Chest X-Ray Pneumonia Classification

This project demonstrates the development, evaluation, and deployment of deep learning models for classifying chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

The project uses **ResNet18** and **DenseNet121**, two convolutional neural network architectures pretrained on ImageNet and adapted through transfer learning for binary chest X-ray classification.

An interactive **Streamlit web application** allows users to upload chest X-ray images, generate predictions using both models, and compare their classification results.

This project was developed as part of the **AIML-501: Model Development** course residency at Indiana Wesleyan University.

## 1. Project Objectives

The primary objectives are to:

- Apply transfer learning using pretrained deep learning architectures.
- Develop binary image classification models for chest X-rays.
- Evaluate and compare ResNet18 and DenseNet121.
- Analyze classification performance using multiple evaluation metrics.
- Build an interactive web application using Streamlit.
- Deploy the application through Streamlit Community Cloud.
- Investigate model generalization, robustness, and limitations.

## 2. Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)

**Source:** [Kaggle — Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset contains chest X-ray images organized into two categories:

- **NORMAL:** Chest X-rays labeled as normal.
- **PNEUMONIA:** Chest X-rays labeled as pneumonia.

The original dataset includes training, validation, and testing directories.

For this project, the training data was divided into training and validation subsets, while the original test set was retained for final model evaluation.

### Data Preprocessing

The image preprocessing pipeline includes:

- Resizing images to **224 × 224 pixels**.
- Converting images to RGB format.
- Applying ImageNet normalization.
- Preparing images as PyTorch tensors.

Class-weighted cross-entropy loss was used during training to help address class imbalance.

## 3. Deep Learning Models

Two pretrained convolutional neural network architectures were evaluated.

### ResNet18

ResNet18 is a residual neural network that uses skip connections to support effective training of deep networks.

For this project:

- A pretrained ResNet18 architecture was used.
- Its original classification layer was replaced with a two-class output layer.
- The model was adapted to classify NORMAL and PNEUMONIA chest X-rays.

### DenseNet121

DenseNet121 uses dense connections between layers to promote feature reuse and improve information flow.

For this project:

- A pretrained DenseNet121 architecture was used.
- Its original classification layer was replaced with a two-class output layer.
- The model was adapted to classify NORMAL and PNEUMONIA chest X-rays.

Both models were trained using PyTorch and TorchVision.

## 4. Model Evaluation

The models were evaluated using several classification metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall classification correctness |
| Precision | Measures the proportion of predicted pneumonia cases that are actually pneumonia |
| Recall (Sensitivity) | Measures the proportion of actual pneumonia cases correctly identified |
| Specificity | Measures the proportion of normal cases correctly identified |
| F1-Score | Balances precision and recall |
| ROC-AUC | Measures the model's ability to distinguish between classes across classification thresholds |
| Confusion Matrix | Displays true positives, true negatives, false positives, and false negatives |

### Healthcare Considerations

In pneumonia classification, false negatives are particularly important because an actual pneumonia case may be incorrectly classified as NORMAL.

False positives also matter because they can lead to unnecessary follow-up testing and concern.

Therefore, model evaluation should consider sensitivity, specificity, and the consequences of classification errors rather than relying on accuracy alone.

## 5. Streamlit Web Application

The project includes an interactive web application developed using Streamlit.

### Application Features

Users can:

1. Upload a chest X-ray image.
2. Preview the uploaded image.
3. Generate predictions using ResNet18 and DenseNet121.
4. View the predicted class from each model.
5. Compare the models' output probabilities.
6. Observe differences between the models' predictions.

The application supports educational demonstrations of deep learning inference and model comparison.

**Important:** The displayed output probabilities are model-generated scores and should not be interpreted as clinically validated disease probabilities.

## 6. Project Structure

```text
AI-Pneumonia-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── saved_models/
    ├── resnet18_pneumonia.pth
    └── densenet121_pneumonia.pth
```

The model files contain the trained weights used by the Streamlit application.

## 7. Technologies Used

- **Python:** Programming language
- **PyTorch:** Deep learning framework
- **TorchVision:** Pretrained model architectures and image transformations
- **ResNet18:** Image classification architecture
- **DenseNet121:** Image classification architecture
- **Streamlit:** Interactive web application framework
- **Google Colab:** Cloud-based development and training environment
- **GitHub:** Version control and project hosting
- **Streamlit Community Cloud:** Web application deployment

## 8. Running the Application Locally

### Step 1 — Clone the Repository

```bash
git clone https://github.com/vinsic2024/AI-Pneumonia-Classification.git
```

### Step 2 — Open the Project Directory

```bash
cd AI-Pneumonia-Classification
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Verify Model Files

Ensure the trained ResNet18 and DenseNet121 model files are available in the locations expected by `app.py`.

### Step 5 — Launch Streamlit

```bash
streamlit run app.py
```

The application should open in a browser, where users can upload images and compare predictions.

## 9. Model Generalization and Robustness

An additional objective of this project is to investigate how the models behave when presented with unfamiliar images.

Testing includes:

- NORMAL chest X-rays from the original test dataset.
- PNEUMONIA chest X-rays from the original test dataset.
- Chest X-rays obtained from external educational sources.
- Unrelated images, such as dogs, trees, and cars.

### Out-of-Distribution Testing

Because the models were adapted for binary chest X-ray classification, they may still produce NORMAL or PNEUMONIA predictions when given unrelated images.

For example, a photograph of a dog may receive a high PNEUMONIA output probability even though it is not a chest X-ray.

This demonstrates an important limitation:

**High model output probabilities do not necessarily indicate reliable or meaningful predictions.**

Potential improvements include input validation, out-of-distribution detection, confidence calibration, and evaluation on independent external datasets.

## 10. Limitations

Several limitations should be considered:

- The models were trained using a limited chest X-ray dataset.
- Class imbalance may influence model predictions.
- Performance may vary across imaging equipment and patient populations.
- The application does not currently establish whether an uploaded image is a valid chest X-ray.
- Predictions on unfamiliar images may be unreliable.
- The models have not been clinically validated.

Further evaluation would be necessary before considering any real-world clinical application.

## 11. Future Improvements

Potential enhancements include:

- Testing on larger, independent chest X-ray datasets.
- Improving sensitivity and specificity.
- Implementing out-of-distribution detection.
- Adding image validation before classification.
- Evaluating probability calibration.
- Incorporating explainable AI methods such as Grad-CAM.
- Improving application usability and error handling.

## 12. Educational Purpose and Disclaimer

**This project is intended strictly for educational and research demonstration purposes.**

The application is not a clinically validated medical diagnostic system.

It must not be used for patient diagnosis, treatment decisions, or other clinical decision-making.

Only appropriately licensed educational images without identifiable patient information should be uploaded.

## 13. Author and Course Information

**Vincent Sichula, Ph.D.**

Professor of Chemistry and Data Scientist

**Course:** AIML-501 — Model Development

**Institution:** Indiana Wesleyan University

**Project:** Chest X-Ray Pneumonia Classification Using Transfer Learning and Streamlit Deployment

## 14. Acknowledgments

- Indiana Wesleyan University
- PyTorch and TorchVision development communities
- Streamlit development community
- Kaggle Chest X-Ray Pneumonia Dataset contributors

---

**Project Summary:** This project demonstrates the complete machine learning workflow—from data preparation and transfer learning to model evaluation, application development, deployment, and robustness testing.
