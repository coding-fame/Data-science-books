# **NLP: Replacing and Correcting Words**

Text preprocessing often involves converting, removing, and correcting various aspects of text data to make it more suitable for analysis.

---

## **1. Text Conversions**
Text can be converted between uppercase and lowercase for normalization.

```python
# Converting lowercase to uppercase
text = "hello good morning"
print(text.upper())

# Converting uppercase to lowercase
text = "HELLO GOOD MORNING"
print(text.lower())
```

---

## **2. Removing Numbers**
Numbers can be removed from text using regular expressions.

```python
import re 

myString = 'Box A has 4 red and 6 white balls, while Box B has 3 red and 5 blue balls.' 
output = re.sub(r'\d+', '', myString) 
print(output)
```

---

## **3. Removing Punctuations**
We can use regular expressions to remove punctuations from text.

```python
import re 

text = "Hello $@#$# Good !@#!@# morning #*#@&@#"
print("Original Text:", text) 

res = re.sub(r'[^\w\s]', '', text ) 
print("Cleaned Text:", res)
```

---

## **4. Removing Whitespaces**
We can remove extra whitespaces using the `strip()` method.

```python
text = "           a sample string     " 
print("Before Stripping:", text)

res = text.strip() 
print("After Stripping:", res)
```

---

## **5. Part of Speech (POS) Tagging**
Assigns parts of speech (nouns, verbs, adjectives, etc.) to words based on context.

### **Install `textblob` Library**
```sh
pip install textblob
```

### **POS Tagging Example**
```python
from textblob import TextBlob 
import nltk 

nltk.download('averaged_perceptron_tagger') 

myString = "Parts of speech: an article, to run, fascinating, quickly, and, of" 
output = TextBlob(myString)
print(output.tags)
```

### **POS using WordNet**
```python
from nltk.corpus import wordnet 

syn = wordnet.synsets('hello')[0] 
print("Syn tag:", syn.pos()) 
 
syn = wordnet.synsets('doing')[0] 
print("Syn tag:", syn.pos()) 
 
syn = wordnet.synsets('beautiful')[0] 
print("Syn tag:", syn.pos()) 
```

---

## **6. Information Extraction**
Information extraction (IE) involves extracting structured information from unstructured text.

Information extraction is useful for:
- Business intelligence
- Resume harvesting
- Media analysis
- Sentiment detection
- Patent search
- Email scanning

---

## **7. Collocations: Bigrams and Trigrams**

### **What is Collocation?**
- Collocations are word pairs frequently occurring together in a paragraph.
- They help in feature extraction, particularly in sentiment analysis.
- Two types of collocations:
  - **Bigrams:** Combination of two words
  - **Trigrams:** Combination of three words

### **Bigram Example**
```python
import nltk
text = "Data Science is a totally new kind of learning experience."
Tokens = nltk.word_tokenize(text)
output = list(nltk.bigrams(Tokens))
print(output)
```

### **Trigram Example**
```python
import nltk
text = "Data Science is a totally new kind of learning experience."
Tokens = nltk.word_tokenize(text)
output = list(nltk.trigrams(Tokens))
print(output)
```

---

## **8. WordNet**
WordNet is an NLP lexical database for English that helps find meanings, synonyms, and antonyms.

### **Synset**
Synset is a simple interface in NLTK for WordNet lookup. Synsets group synonymous words expressing the same concept.

### **WordNet Examples**
```python
from nltk.corpus import wordnet 

syn = wordnet.synsets('hello')[0] 
print("Synset name:", syn.name()) 
print("Synset meaning:", syn.definition()) 
print("Synset example:", syn.examples())
```

```python
syn = wordnet.synsets('boy')[0] 
print("Synset name:", syn.name()) 
print("Synset meaning:", syn.definition()) 
print("Synset example:", syn.examples())
```

```python
syn = wordnet.synsets('good')[0] 
print("Synset name:", syn.name()) 
print("Synset meaning:", syn.definition()) 
print("Synset example:", syn.examples())
```

---

## Conclusion

Text preprocessing in NLP involves a variety of tasks, including converting case, removing numbers and punctuation, stripping whitespace, and tagging parts of speech. Using tools like **TextBlob**, **NLTK**, and **WordNet**, we can extract valuable insights and prepare text for further analysis. Collocations, bigrams, and trigrams are particularly useful for tasks like sentiment analysis and information extraction.

