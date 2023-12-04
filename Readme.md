# Sentiment Analysis on IMDb Movie Reviews

## Overview

This project performs sentiment analysis on a dataset of IMDb movie reviews. The goal is to classify reviews into positive or negative sentiments using machine learning models. The analysis includes data preprocessing, text vectorization, model training, evaluation, and visualization of key features.

## Project Structure

The project is organized as follows:

- **Code Files:**
  - `sentiment_analysis.ipynb`: Jupyter Notebook containing the complete code for sentiment analysis.
  - `IMDB_Dataset/IMDB Dataset.csv`: The dataset used for training and testing.

- **Libraries Used:**
  - NumPy, Pandas, Seaborn, Matplotlib, NLTK, BeautifulSoup, Scikit-learn, Spacy, TextBlob, WordCloud.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/SakshiYeole/Sentiment-Analysis-of-IMDB-Movie-Reviews.git

2. **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook sentiment_analysis.ipynb

This will open the Jupyter Notebook in your default web browser. Navigate to the 'sentiment_analysis.ipynb' file.

3. **Explore the Code:**
- Data Preprocessing:
    - Load the IMDb dataset and perform exploratory data analysis.
    - Split the dataset into training and testing sets.

- Text Preprocessing:
    - Apply various text cleaning techniques, including HTML tag removal and stopword removal.
    - Normalize the text data through stemming.

- Text Vectorization:
    - Utilize Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) to convert text data into numerical features.

- Sentiment Labeling:
    - Use LabelBinarizer to transform sentiment labels into binary format.

- Model Training:
    - Train three machine learning models: Logistic Regression, Stochastic Gradient Descent with SVM, and Multinomial Naive Bayes.
    - Train models separately using BoW and TF-IDF features.

- Model Evaluation:
    - Evaluate model performance using accuracy scores, classification reports, and confusion matrices.

- Word Cloud Visualization:
    - Visualize Word Clouds for positive and negative reviews to highlight key words.

## Results and Visualizations:
- Explore accuracy scores, classification reports, and confusion matrices to understand the models' performance.
- Analyze Word Clouds to identify prominent words in positive and negative reviews.
