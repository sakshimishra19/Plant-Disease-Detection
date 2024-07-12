# Plant Disease Detection using Deep Learning

## Project Definition
This project uses Convolutional Neural Networks (CNNs) to detect plant diseases from leaf images, aiding in early identification and management of crop health issues.

## Dataset
Dataset Link: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Project Steps
This project was completed in 6 main steps:

1. Data Preparation:
   - Importing and preprocessing the dataset
   - Splitting into training and validation sets

2. Model Building:
   - Designing a CNN architecture
   - Adding convolutional, pooling, and dense layers

3. Training:
   - Compiling the model with appropriate optimizer and loss function
   - Training the model for 10 epochs

4. Evaluation:
   - Assessing model performance on training and validation sets
   - Generating confusion matrix and classification report

5. Model Saving:
   - Saving the trained model for future use

6. Testing:
   - Performing predictions on new, unseen images

## Key Features
- Data preprocessing and augmentation
- CNN-based deep learning model
- Visualization of training and validation accuracy
- Confusion matrix for detailed performance analysis
- Single image prediction capability

## Requirements
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Scikit-learn

## Usage
1. Clone the repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python script
4. For single image prediction, use the provided function with the path to your image

## Results
- The model achieves high accuracy on both training and validation sets
- Detailed performance metrics are provided through confusion matrix and classification report

