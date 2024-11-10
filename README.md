# Alpaca Classifier with MobileNetV2 Transfer Learning
This is one of AndrewNg Programming Assignments from the DL Specialization

This task demonstrates how to adapt the MobileNetV2 model, pre-trained on the ImageNet dataset, for a custom image classification task. Using transfer learning techniques, the model was fine-tuned for improved accuracy on a specific dataset.

## Project Overview

MobileNetV2, a lightweight and efficient CNN model, was employed for this project. Originally trained on ImageNet (over 14 million images across 1000 classes), MobileNetV2 serves as the feature extractor. We adapted the model using TensorFlow's Functional API, allowing for flexible fine-tuning to classify images within a new dataset.


![image](https://github.com/user-attachments/assets/68fc6a90-8dfa-4e4b-a567-4ac75ea8ec80)

## Key Objectives

- Load and structure a dataset from a directory.
- Preprocess and augment data using TensorFlow's `Sequential` API.
- Customize and train MobileNetV2 on a new dataset using the Functional API.
- Fine-tune the final layers of the classifier for enhanced accuracy.

## Implementation Details

### 1. Data Loading and Preprocessing
- **Dataset Creation**: Images were organized into directories for each class, allowing `ImageDataGenerator` to load and label them automatically.
- **Data Augmentation**: Applied transformations like rotation, zoom, horizontal flip, and scaling to increase dataset diversity and improve model robustness.
- **Batching and Shuffling**: The dataset was divided into training and validation sets with batching and shuffling to optimize training and prevent overfitting.

### 2. Model Customization
- **MobileNetV2 as Base Model**: Initialized MobileNetV2 with pre-trained weights from ImageNet and excluded the top layers.
- **New Classification Layers**: Added custom dense layers and dropout to tailor MobileNetV2 for our dataset. This allowed the model to learn from the new classes without changing its efficient architecture.
- **Functional API**: Used TensorFlow’s Functional API to combine the base MobileNetV2 model with new layers, maintaining flexibility for adjustments.

### 3. Training and Fine-Tuning
- **Initial Training**: Trained only the new classification layers with frozen base layers to allow MobileNetV2 to act as a feature extractor.
- **Fine-Tuning**: Unfrozen the top layers of MobileNetV2 for further training to adapt more specific features to the new dataset.
- **Optimization**: Used Adam optimizer and reduced learning rate during fine-tuning to achieve smoother convergence and prevent drastic changes to pre-trained weights.
- **Metrics**: Tracked accuracy and loss throughout training and validation to evaluate and fine-tune model performance.

  ## Results

- **Accuracy**: The model achieved high accuracy on the validation set after fine-tuning, demonstrating effective feature extraction and adaptation with MobileNetV2.
- **Efficiency**: With MobileNetV2's lightweight architecture, the model provided fast inference, making it suitable for real-time or resource-constrained applications.
- **Transfer Learning Impact**: Leveraging a pre-trained model significantly reduced training time and improved convergence, allowing the model to reach satisfactory performance with a smaller dataset.

## Conclusion

This task highlighted the effectiveness of transfer learning using MobileNetV2 for image classification. By fine-tuning the model’s final layers, we achieved competitive accuracy while maintaining computational efficiency.
