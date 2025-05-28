# Presidential Speech Analysis

Analyzes and visualizes the sentiment and readability of U.S. presidential speeches over time using NLP and machine learning techniques.

---

## 🔍 Overview

This project processes presidential speeches to extract sentence-level sentiment scores, various readability metrics, and lexical diversity. It then aggregates these metrics by president and applies Principal Component Analysis (PCA) to visualize speech characteristics. Finally, it provides interactive plots showing sentiment trends over time.

- **Problem solved:** Enables comparative analysis of presidential speech complexity and sentiment to reveal communication styles and historical patterns.
- **Audience:** Data scientists, political analysts, historians, and NLP enthusiasts.
- **Purpose:** A learning exercise exploring text processing, sentiment analysis, readability metrics, and dimensionality reduction in a real-world context.

> This project uses the NLTK VADER sentiment analyzer, the textstat readability library, and scikit-learn’s PCA to extract insights from cleaned presidential speech transcripts as well as bertopic.

---

## 🛠️ Tech Stack

- Python
- pandas, NumPy  
- NLTK (sent_tokenize, word_tokenize, VADER)  
- textstat (readability metrics)  
- scikit-learn (PCA, StandardScaler)  
- matplotlib (static PCA biplot)  
- Plotly Express (interactive sentiment timeline)  
- Bertopic
- Marimo

---

## 🚀 Features

- Sentence-level sentiment scoring using VADER  
- Comprehensive readability metrics including Flesch Reading Ease, SMOG, Coleman-Liau, Dale-Chall, and lexical diversity  
- Aggregation of speech metrics by president for comparative analysis  
- PCA biplot to visualize multivariate speech features and their relationships  
- Interactive Plotly visualization of speech sentiment over time, colored by president and party

---

## 📁 Project Structure

```
├── speeches.json # Transcripts
├── presidents.csv # Presidents metadata
├── pres_popularity.csv # Presidents metadata
├── requirements.txt # Required Python packages
├── README.md # This file
```

---

## 📈 Results

- Presidents vary significantly in readability and sentiment profiles, with PCA revealing clusters related to political party and speech complexity.  
- Example insight: Presidents with more readable speeches tend to have lower approval ratings, a nearly antiparallel relationship seen in PCA loadings.  
- Interactive Plotly timeline helps correlate speech sentiment fluctuations with historical contexts.

---

## 🧠 What I Learned

- Practical use of NLP for sentiment analysis at the sentence level within longer documents  
- Application of multiple readability metrics and custom lexical diversity calculations  
- Effective dimensionality reduction with PCA for high-dimensional text features  
- Visualization techniques combining matplotlib and Plotly for static and interactive plots  
- Data merging and aggregation strategies to enrich text data with metadata

---

## 📦 Installation & Usage

```bash
git clone https://github.com/NathanielH-snek/PresidentialSpeechAnalysis.git
cd PresidentialSpeechAnalysis
pip install -r requirements.txt
marimo run notebook.py
```