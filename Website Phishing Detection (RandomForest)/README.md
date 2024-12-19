
# Website Phishing Detection

This project implements a machine learning system to detect phishing websites. The project explores data preprocessing, feature engineering, and machine learning techniques to achieve this goal.

## Dataset

The dataset used for this project contains 48 features extracted from 5,000 phishing webpages and 5,000 legitimate webpages, collected between January to May 2015 and May to June 2017. The features were extracted using the Selenium WebDriver for precise and robust data collection.

### Dataset Statistics:
- The dataset is balanced, with an equal number of phishing and legitimate webpages.
- Various features include counts of special characters, URL lengths, and presence of certain elements.

## Steps

### 1. Dataset Analysis
- **Visualization**: Used histograms to analyze the distribution of the `CLASS_LABEL` feature.
- **Descriptive Statistics**: Summarized key statistics for numerical features.

### 2. Data Preprocessing
- Dropped the `id` column as it is not relevant for model training.
- Split the data into training and testing sets (80% training, 20% testing).

### 3. Model Training
Implemented a **Random Forest** classifier:
- Trained using 80% of the dataset.
- Tested on 20%, achieving an accuracy of **98.4%**.

### 4. Model Evaluation
- Evaluated the model using accuracy, precision, and recall metrics.
- Achieved a precision of **98.23%** and a recall of **98.62%**.

## Prerequisites

### Python Libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`

### Dataset
The dataset should be available in the `dataset/` folder with the file name `Phishing_Legitimate_full 2.csv`.

## Results

| Metric          | Value    |
|------------------|----------|
| Accuracy         | 98.4%   |
| Precision        | 98.23%  |
| Recall           | 98.62%  |
