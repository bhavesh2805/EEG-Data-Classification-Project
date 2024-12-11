
EEG Data Classification Project
Overview
This project involves building a classification model to analyze and classify EEG data, which is widely used in neuroscience and medical fields, including epilepsy diagnosis. The project uses two EEG datasets to train and evaluate the model.
Table of Contents
Objective
Tasks Overview
Installation
Usage
Features
Results and Visualizations
Technologies Used
Contributing
License
Acknowledgments
Objective
The primary goal of this project is to preprocess, analyze, and classify EEG data by leveraging advanced machine learning models. The results aim to improve understanding and applications such as epilepsy diagnosis.
Tasks Overview
The project is divided into several key tasks:
Data Preprocessing: Cleaning and filtering EEG data for analysis.
Feature Extraction: Extracting meaningful features using Recurrence Quantification Analysis and Recurrence Network Analysis.
Data Splitting: Dividing the dataset into training, validation, and testing subsets.
Model Selection: Experimenting with CNN, LSTM, and hybrid CNN-LSTM architectures.
Model Training: Training the models and evaluating their performance.
Model Evaluation: Assessing the models using precision, recall, F1-score, and accuracy.
Testing: Evaluating the model on unseen data.
Results and Visualization: Analyzing performance metrics and visualizations to understand model outcomes.
Installation
Clone the repository:
git clone https://github.com/your-username/eeg-classification.git
Navigate to the project directory:
cd eeg-classification
Install dependencies:
pip install -r requirements.txt
Usage
Preprocess the EEG data:
python preprocess.py
Extract features:
python feature_extraction.py
Train the model:
python train_model.py
Evaluate the model:
python evaluate.py
Features
Preprocessing: Bandpass filtering to retain critical frequencies (1-50 Hz).
Feature Extraction: Recurrence Quantification Analysis and Recurrence Network Analysis.
Model Architectures:
Convolutional Neural Network (CNN)
Long Short-Term Memory (LSTM) Network
Hybrid CNN-LSTM
Performance Metrics: Precision, recall, F1-score, accuracy, confusion matrix, and ROC curve.
Results and Visualizations
Key Metrics:
Overall accuracy: ~53.50%
Class 0 precision: 52%, recall: 83%, F1-score: 64%
Class 1 precision: 60%, recall: 25%, F1-score: 35%
Visualizations:
Confusion matrix
ROC curve
Precision-recall curve
Recommendations for Improvement
Address class imbalance using techniques like oversampling or class weighting.
Experiment with alternative models like Random Forest or Gradient Boosting.
Perform hyperparameter tuning to optimize model performance.
Use ensemble methods or transfer learning to enhance accuracy.
Technologies Used
Programming Language: Python
Libraries and Tools:
NumPy, Pandas, Scikit-learn, TensorFlow/Keras
Matplotlib, Seaborn
License
This project is licensed under the MIT License.
Acknowledgments
Special thanks to Rohan Verma and Bhavesh Kulkarni for their contributions to this project as part of the IE6400 Foundations of Data Analytics Engineering course at Northeastern University.

