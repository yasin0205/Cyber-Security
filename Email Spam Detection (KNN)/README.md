
# Email Spam Detection

This project implements a machine learning system to classify email messages as either **spam** or **ham** (non-spam). The project explores text processing, feature engineering, and machine learning techniques to achieve this goal.

## Dataset

The dataset used for this project contains 5,171 email messages, with the following columns:

- `Unnamed`: An index or identifier for each email.
- `label`: The class of the email (`spam` or `ham`).
- `text`: The content of the email message.
- `label_num`: A numeric encoding of the label (`0` for ham, `1` for spam).

### Dataset Statistics:
- Class distribution is imbalanced, with more **ham** messages than **spam**.
- Text length varies significantly between classes:
  - **Ham**: Average length ~977 characters
  - **Spam**: Average length ~1,223 characters

## Steps

### 1. Exploratory Data Analysis (EDA)
- **Visualization**: Used bar plots to analyze class distribution.
- **Text Analysis**:
  - Frequent word occurrence for spam and ham messages.
  - Length distribution across messages.

### 2. Data Preprocessing
- Removed punctuation and stopwords.
- Tokenized and cleaned text for feature extraction.

### 3. Feature Engineering
- Used Bag of Words (BoW) model to transform text into numerical vectors.
- Produced a sparse matrix with 50,179 unique features.

### 4. Model Training
Implemented a **K-Nearest Neighbors (KNN)** classifier:
- Trained using 80% of the dataset.
- Tested on 20%, achieving an accuracy of **83.28%**.

### 5. Optional OpenAI ChatGPT Classifier
Integrated an experimental version using OpenAI's GPT-4 for email classification. This approach can classify email content as **spam** or **not spam** but requires API access.

## Prerequisites

### Python Libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`
- `openai` (optional for ChatGPT integration)



### Dataset
The dataset should be available in the `dataset/` folder with the file name `spam_ham_dataset.csv`.



## Results

| Metric          | Value    |
|------------------|----------|
| Accuracy         | 83.28%  |

