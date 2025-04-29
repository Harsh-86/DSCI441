# DSCI441
Sentiment Analysis
This project focuses on building a robust sentiment analysis model to classify social media posts as positive, negative, or neutral. The study employs natural language processing (NLP) techniques like tokenization, stopword removal, and emoji/hashtag handling to preprocess raw text. Traditional machine learning models (e.g., logistic regression with TF-IDF features) are compared against deep learning architectures (e.g., LSTMs, BERT) to evaluate performance on noisy, real-world text data. Key challenges include handling sarcasm, slang, and context-dependent sentiments.

# BERT-based Sentiment Analysis on Tweets

This project applies a fine-tuned BERT model to perform **sentiment analysis** on a dataset of tweets.  
The objective is to classify each tweet as either **positive** or **negative** based on its text content.

## 📚 Overview

- **Model**: BERT (`bert-base-uncased`) from Huggingface Transformers.
- **Task**: Binary text classification (Sentiment Analysis).
- **Dataset**: CSV file containing tweets labeled as 0 (negative) or 1 (positive).
- **Frameworks**: PyTorch, Huggingface Transformers.

## 📋 Steps Performed

1. **Data Loading & Preprocessing**  
   - Read dataset from CSV.
   - Clean and prepare text and labels for model input.

2. **Tokenization**  
   - Use `BertTokenizer` to tokenize the sentences into input IDs and attention masks.

3. **Data Preparation**  
   - Create `TensorDataset` and `DataLoader` for efficient mini-batch training and evaluation.

4. **Model Fine-tuning**  
   - Load a pre-trained `BertForSequenceClassification`.
   - Fine-tune on the tweets dataset using the AdamW optimizer and a learning rate scheduler.

5. **Evaluation**  
   - Measure model performance using accuracy, confusion matrix, and classification report.
   - Plot the training loss over epochs to monitor convergence.'

## 📈 Results
   - The model was able to achieve high accuracy on the sentiment classification task.
   - Training and validation losses were plotted to observe model performance over epochs.

## 📂 File Structure
   - BERT.ipynb — Main Jupyter Notebook containing all the code for loading data, preprocessing, model training, and evaluation.
   - data.csv — Dataset used (you’ll need to provide this file in the correct path).


## ✨ Future Improvements
   - Hyperparameter tuning (learning rate, batch size).
   - Incorporating more advanced techniques like learning rate warm-up.
   - Exploring larger pre-trained models such as bert-large-uncased.
   - Data augmentation or synthetic data generation for better generalization.
