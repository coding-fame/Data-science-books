# **NLP: Bag of Words, TF, and IDF**

In Natural Language Processing (NLP), various methods and techniques are employed to analyze and process raw text. Two fundamental concepts in NLP are the **Bag of Words (BoW)** model and **Term Frequency-Inverse Document Frequency (TF-IDF)**. These models help convert text into a numerical format, making it easier to apply machine learning algorithms for text analysis.

## **Bag-of-Words (BoW)**

### **Overview**
- Bag-of-Words is a method of extracting essential features from raw text.
- It converts raw text into individual words and counts the frequency of each word.
- Useful for machine learning models that process textual data.

### **Steps in Bag-of-Words**
1. **Raw Text**
   - This is the original text that needs analysis.

2. **Clean Text**
   - Remove unnecessary data like punctuation marks and stopwords.

3. **Tokenization**
   - Convert sentences into individual words (tokens).

4. **Building Vocabulary**
   - Identify all unique words from the text after preprocessing.

5. **Generate Word Frequencies**
   - Count the occurrences of each word in the text.

### **Example Use Case**
#### Given Sentences:
1. "Jim and Pam travelled by bus."
2. "The train was late."
3. "The flight was full. Travelling by flight is expensive."

#### **Bag-of-Words Implementation in Python**
```python
from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "Jim and Pam travelled by bus.",
    "The train was late",
    "The flight was full. Travelling by flight is expensive"
]

cv = CountVectorizer()
B_O_W = cv.fit_transform(sentences).toarray()
print(B_O_W)
```

### **Applications of Bag-of-Words**
- Information retrieval from documents
- Text classification

### **Limitations of Bag-of-Words**
- **Semantic meaning:** Does not consider the meaning of words in context.
- **Vector size:** Large documents result in higher computational costs.
- **Preprocessing required:** Data must be cleaned before use.

---

## **Term Frequency-Inverse Document Frequency (TF-IDF)**

### **Overview**
- **TF-IDF** is combines two metrics: TF and IDF
- **Term Frequency (TF):** Measures how frequently a term appears in a document.
- **Inverse Document Frequency (IDF):** Measures how important a term is across multiple documents.
- Used in search engines, document ranking, and text mining.

### **Example Dataset - Movie Reviews**
#### Given Reviews:
1. "This movie is very scary and long."
2. "This movie is not scary and is slow."
3. "This movie is spooky and good."

### **Step 1: Calculating Term Frequency (TF)**
TF is calculated as:
\[ TF(t) = \frac{\text{Number of times term t appears in a document}}{\text{Total terms in the document}} \]

### **Step 2: Calculating Inverse Document Frequency (IDF)**
IDF measures how important a term is across all documents:
\[ IDF(t) = \log{\left( \frac{N}{df} \right)} \]
Where:
- **N** = Total number of documents
- **df** = Number of documents containing the term

#### **Example Calculation for Review 2**
- IDF(‘movie’) = log(3/3) = 0
- IDF(‘is’) = log(3/3) = 0
- IDF(‘not’) = log(3/1) = 0.48
- IDF(‘scary’) = log(3/2) = 0.18
- IDF(‘and’) = log(3/3) = 0
- IDF(‘slow’) = log(3/1) = 0.48

### **Step 3: Calculating Final TF-IDF Values**
TF-IDF score for each word:
\[ TF-IDF = TF \times IDF \]

#### **Example Calculation for Word "this" in Review 2**
- TF-IDF('this') = TF('this') * IDF('this') = \( \frac{1}{8} \times 0 = 0 \)

---

### **Summary**
- **Bag-of-Words** captures word frequency but ignores context.
- **TF-IDF** refines this by emphasizing important words.
- These techniques are widely used in text processing tasks such as sentiment analysis, spam detection, and document classification.

---
