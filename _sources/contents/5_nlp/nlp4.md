# NLP: Components in NLP

## Common NLP Libraries
- **NLTK (Natural Language Toolkit)**
- **spaCy**
- **TextBlob**
- **Gensim**
- **Pattern**

## Components in NLP
NLP consists of five main components:
1. **Lexical Analysis**
2. **Syntactic Analysis**
3. **Semantic Analysis**
4. **Discourse Analysis**
5. **Pragmatic Analysis**

---

### 1. Lexical Analysis
- Lexical refers to words or vocabulary of a language.
- Involves dividing a text into:
  - **Paragraphs**
  - **Sentences**
  - **Words**
- It helps in identifying and analyzing word structures.

### 2. Syntactic Analysis
- Focuses on **sentence structure** and **grammar**.
- Analyzes how words are arranged to form meaningful sentences.
- **Example:**
  - "The shop goes to the house" → ❌ Invalid (Incorrect syntax)

### 3. Semantic Analysis
- Focuses on **meaning** in language.
- Ensures sentences make sense.
- **Example:**
  - "Hot ice cream" → ❌ Invalid (Contradictory meaning)

### 4. Discourse Analysis
- Considers the meaning **beyond** individual sentences.
- **Example:**
  - "He works at Google" → "He" must refer to a previously mentioned person.

### 5. Pragmatic Analysis
- Focuses on **context** and the practical use of language.
- Determines meaning based on different situations.

---

## NLP Use Case: Processing Textual Information

### Given Text File: `story_input.txt`
```text
Once upon a time there was an old mother pig that had three little pigs and not enough food to feed them...
```

### 1. Reading Text from a File
```python
with open("story_input.txt", 'r') as f:
    text = f.read()
    print(text)
    print(type(text))
    print(len(text))
```

### 2. Sentence Tokenization
```python
import nltk
from nltk import sent_tokenize

data = open("story_input.txt").read()
sentences = sent_tokenize(data)

print("Total sentences:", len(sentences))
for sent in sentences:
    print(sent)
```

### 3. Word Tokenization
```python
import nltk
from nltk import word_tokenize

data = open("story_input.txt").read()
words = word_tokenize(data)
print(words)
```

### 4. Finding Word Frequency
```python
from nltk.probability import FreqDist

data = open("story_input.txt").read()
words = word_tokenize(data)
fdist = FreqDist(words)
print(fdist.most_common(10))
```

### 5. Plotting Word Frequency Graph (Including Punctuation)
```python
import matplotlib.pyplot as plt
from nltk.probability import FreqDist

data = open("story_input.txt").read()
words = word_tokenize(data)
fdist = FreqDist(words)
fdist.plot(10)
```

### 6. Removing Punctuation Marks
```python
words_no_punc = [w.lower() for w in words if w.isalpha()]
print(words_no_punc)
```

### 7. Plotting Word Frequency Graph (Without Punctuation)
```python
fdist = FreqDist(words_no_punc)
fdist.plot(10)
```

### 8. Removing Stop Words
```python
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
clean_words = [w for w in words_no_punc if w not in stopwords]
print(clean_words)
```

### 9. Final Word Frequency Distribution
```python
fdist = FreqDist(clean_words)
fdist.plot(10)
```

---

## Word Cloud - Data Visualization
- Word Cloud visually represents the most frequent words in a text.

```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

text = "Python is good programming language, Python is very easy, Learning Data Science starts from Python"
wordcloud = WordCloud().generate(text)

plt.figure(figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
```

---
### Summary
- **Lexical, Syntactic, Semantic, Discourse, and Pragmatic Analysis** are key NLP components.
- **Tokenization, stop-word removal, and word frequency analysis** are essential steps in text processing.
- **Word Clouds** help visualize word importance.


