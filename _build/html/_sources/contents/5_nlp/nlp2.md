# NLP: Text Wrangling and Cleaning

## Introduction
Text wrangling and cleaning are essential steps in Natural Language Processing (NLP) to ensure that machines can understand text data effectively. These steps help in preprocessing text for better insights and model performance.

### Key Steps in Text Processing:
- **Tokenization**
- **Lemmatization**
- **Stemming**
- **Stop Word Removal**

---
## 1. Downloading `punkt`
One common tool for tokenization is the **Punkt** tokenizer.
```python
import nltk
nltk.download("punkt")
```
The `punkt` tokenizer is used to split text into sentences. Let's move forward with text wrangling.

---
## 2. Sentence Splitting
A paragraph consists of multiple sentences. Splitting paragraphs into sentences helps with better text understanding.

### **Using `split()` Method**
```python
myString = "This is a paragraph. It should split at the end of sentence marker, such as a period. It can tell that the period in Mr.Daniel is not an end. Run it!, Hey How are you doing"
result = myString.split(".")
```

### **Using `sent_tokenize()` Function**
```python
from nltk.tokenize import sent_tokenize

tokenized_sentence = sent_tokenize(myString)
print(tokenized_sentence)
```
#### **Note:**
- The paragraph is correctly split into sentences.
- It differentiates between a period that ends a sentence and one used in a name (e.g., `Mr. Daniel`).

---
## 3. Tokenization
Tokenization is the process of breaking text into smaller units called tokens. These can be words or sentences.

### **Types of Tokenization**
- **Sentence Tokenization**: Every sentence is identified as a token.
- **Word Tokenization**: Every word is identified as a token.

### **Word Tokenization Methods**
```python
# 1. Using `split()` Method
myString = "These are sentences. Let us tokenize it! Run it!"
print(myString.split())

# 2. Using `word_tokenize()` Function
from nltk.tokenize import word_tokenize
print(word_tokenize(myString))

# 3. Using `regexp_tokenize()` Function
from nltk.tokenize import regexp_tokenize
print(regexp_tokenize(myString, pattern="\w+"))

# Capturing digits from a sentence
myString = "These are 3 sentences. Let us tokenize them! Run the code!"
print(regexp_tokenize(myString, pattern="\d+"))
```

---
## 4. Stemming
Stemming reduces words to their root form by removing suffixes.

### **Example of Stemming**
```python
from nltk.stem import PorterStemmer

porter = PorterStemmer()
print(porter.stem("cutting"))
```

### **Stemming Multiple Words**
```python
e_words = ["wait", "waiting", "waited", "waits"]
ps = PorterStemmer()
for w in e_words:
    print(ps.stem(w))
```

### **Advanced Stemming Example**
```python
import nltk
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()
text = "studies studying cries cry"
tokens = nltk.word_tokenize(text)

for w in tokens:
    print("Stemming for {} is {}".format(w, porter_stemmer.stem(w)))
```
#### **Note:**
- Stemming is simple but may not always be accurate.
- For more precise results, use **Lemmatization**.

---
## 5. Lemmatization
Lemmatization is more advanced than stemming as it considers the context and part of speech.

### **Example of Lemmatization**
```python
from nltk.stem import WordNetLemmatizer

wl = WordNetLemmatizer()
print("rocks :", wl.lemmatize("rocks"))
print("corpora :", wl.lemmatize("corpora"))
print("better :", wl.lemmatize("better", pos="a"))
```

### **Lemmatizing a Sentence**
```python
import nltk
from nltk.stem import WordNetLemmatizer

wl = WordNetLemmatizer()
text = "studies studying cries cry"
tokens = nltk.word_tokenize(text)

for word in tokens:
    print("Lemmatization for {} is {}".format(word, wl.lemmatize(word)))
```

---
## 6. Stop Word Removal
Stop words are common words that do not contribute much to the meaning of a sentence (e.g., "a", "in", "the").

### **List Stop Words in English**
```python
import nltk
from nltk.corpus import stopwords

print(stopwords.words('english'))
```

### **Check Available Languages**
```python
from nltk.corpus import stopwords
langs = stopwords.fileids()
print(len(langs))
```

### **Removing Stop Words from Text**
```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

mylist = stopwords.words('english')
line = "This is really good, how are you doing"
postPa = [word for word in line.split() if word not in mylist]
print(postPa)
```

---

## Conclusion
- **Text wrangling and cleaning** are fundamental in NLP.
- **Tokenization**, **stemming**, **lemmatization**, and **stop word removal** help structure text for machine learning models.
- **NLTK** provides powerful tools to handle these tasks efficiently.

---
