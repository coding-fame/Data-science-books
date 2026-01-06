# **Twitter Sentiment Analysis using TextBlob**

## **Introduction to TextBlob**
TextBlob is a simple API for performing various **Natural Language Processing (NLP)** tasks such as:
- Part-of-Speech Tagging
- Noun Phrase Extraction
- Sentiment Analysis
- Classification (Naive Bayes, Decision Tree)
- Language Translation and Detection
- Spelling Correction

TextBlob is built on top of the **Natural Language Toolkit (NLTK)**.

### **Sentiment Analysis in TextBlob**
- Sentiment Analysis categorizes text into different sentiments such as **positive** or **negative**.
- It can also classify text into **neutral**, **highly positive**, or **highly negative** categories.
- TextBlob assigns two scores:
  - **Polarity**: A float value ranging from **-1.0 to 1.0**, where negative values indicate negative sentiment and positive values indicate positive sentiment.
  - **Subjectivity**: A float value between **0.0 to 1.0**, where **0.0 is objective** and **1.0 is subjective**.

### **Installation**
```bash
pip install -U textblob
```

---
## **Basic Sentiment Analysis using TextBlob**

### Example 1: Positive Sentiment
```python
from textblob import TextBlob

text = TextBlob("It was a wonderful movie. I liked it very much.")
print(text.sentiment)
print(f'Polarity: {text.sentiment.polarity}')
print(f'Subjectivity: {text.sentiment.subjectivity}')
```

### Example 2: Mixed Sentiment
```python
text1 = TextBlob("I liked the acting of the lead actor but I didn't like the movie overall.")
text2 = TextBlob("I liked the acting of the lead actor and I liked the movie overall.")

print(text1.sentiment)
print(text2.sentiment)
```

---
## **Using NLTK’s Twitter Corpus for Sentiment Analysis**

We use **Twitter Samples Corpus** from NLTK to train TextBlob’s Naive Bayes Classifier.

### **Downloading Twitter Sample Datasets**
```python
from nltk.corpus import twitter_samples
import nltk
nltk.download('twitter_samples')

print(twitter_samples.fileids())
```

### **Loading Positive and Negative Tweets**
```python
pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')

print(f'Positive Tweets: {len(pos_tweets)}')
print(f'Negative Tweets: {len(neg_tweets)}')
```

### **Creating Labeled Datasets**
```python
pos_tweets_set = [(tweet, 'pos') for tweet in pos_tweets]
neg_tweets_set = [(tweet, 'neg') for tweet in neg_tweets]
```

### **Splitting Data into Training and Testing Sets**
```python
from random import shuffle 

shuffle(pos_tweets_set)
shuffle(neg_tweets_set)

test_set = pos_tweets_set[:300] + neg_tweets_set[:300]
train_set = pos_tweets_set[300:600] + neg_tweets_set[300:600]
```

## **Training a Naive Bayes Classifier**
```python
from textblob.classifiers import NaiveBayesClassifier

classifier = NaiveBayesClassifier(train_set)
```

### **Testing Model Accuracy**
```python
accuracy = classifier.accuracy(test_set)
print(f'Accuracy: {accuracy:.2f}')
```

### **Predicting Sentiment of Custom Text**
```python
text = "It was a wonderful movie. I liked it very much."
print(classifier.classify(text))

text = "I don't like movies having happy endings."
print(classifier.classify(text))
```

### **Using TextBlob with the Custom Classifier**
```python
blob = TextBlob(text, classifier=classifier)
print(blob.classify())
```

---

## **Conclusion**
- **TextBlob** provides a simple and efficient way to perform Sentiment Analysis.
- It assigns **polarity and subjectivity** scores to text.
- We can train a **Naive Bayes Classifier** using **Twitter Samples Corpus** for better accuracy.
- The trained model can classify new tweets into **positive** or **negative** categories.

🚀 **TextBlob is a great tool for quick NLP implementations!**

---