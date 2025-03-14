# 🚀 Text Summarization (Extractive & Abstractive) using NLP

This repository contains a **Python-based text summarizer** that supports both **extractive** (NLTK-based) and **abstractive** (Transformer-based) summarization. The abstractive summarization runs efficiently on **GPU** using Hugging Face models.

## 📌 Features
- **Extractive Summarization**: Uses **NLTK** to extract key sentences based on word frequency.
- **Abstractive Summarization**: Uses **Hugging Face Transformers (DistilBART)** to generate AI-powered concise summaries.
- **Optimized for GPU**: Uses **CUDA acceleration** for faster processing.
- **Batch Processing Support**: Allows summarization of multiple texts simultaneously.

---
## 🛠 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Real-J/Text-Summarizer-Using-NLP.git
cd Text-Summarizer-Using-NLP
```

### 2️⃣ Install Required Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentencepiece nltk accelerate protobuf
pip install ipywidgets notebook jupyterlab
```

### 3️⃣ Verify GPU Availability
Run the following inside a **Jupyter Notebook** to check if CUDA is detected:
```python
import torch
print("Is CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
```

---
## 🚀 Usage
### **Jupyter Notebook**
Open a **Jupyter Notebook** and run the following:

```python
from summarizer import extractive_summarization, abstractive_summarization

text = """Autonomous vehicles are transforming transportation by reducing human intervention in driving. These vehicles use sensors, cameras, LiDAR, and AI algorithms to navigate. The goal is to enhance road safety and reduce traffic congestion. However, challenges such as unpredictable human behavior, adverse weather conditions, and ethical considerations in decision-making remain. Despite these hurdles, continuous advancements in AI and machine learning are making self-driving technology more reliable and accessible."""

# Extractive Summary
print("Extractive Summary:")
print(extractive_summarization(text))

# Abstractive Summary
print("Abstractive Summary:")
print(abstractive_summarization(text))
```

---
## 🔍 Code Explanation
### 1️⃣ **Extractive Summarization (NLTK-based)**
The extractive summarizer selects the most important sentences from the original text based on **word frequency analysis**.

- **Tokenization**: The text is split into sentences and words.
- **Stopwords Removal**: Common words (e.g., "the", "is", "and") are removed to focus on meaningful words.
- **Word Frequency Calculation**: Each word's occurrence is counted and normalized.
- **Sentence Scoring**: Sentences containing high-frequency words get higher scores.
- **Summary Extraction**: The top `n` sentences with the highest scores are selected.

### 2️⃣ **Abstractive Summarization (Transformer-based)**
Unlike extractive summarization, **abstractive summarization** generates new sentences rather than selecting from the input text. This is done using **Hugging Face Transformers (DistilBART or BART-Large CNN)**.

- **Uses a pre-trained transformer model** fine-tuned for summarization.
- **Encodes the input text** into a high-dimensional representation.
- **Decodes** a shortened summary using attention mechanisms.
- **Leverages GPU acceleration** for faster inference.

---
## ⚡ Performance Optimizations
1. **Faster Summarization with GPU**
   - The script automatically detects and runs on **CUDA (NVIDIA GPUs)**.
   - Ensure **PyTorch is installed with CUDA support** (`torch.cuda.is_available()` should return `True`).

2. **Use a Larger Model for Better Quality** (Optional)
   ```python
   def abstractive_summarization(text, model_name="facebook/bart-large-cnn"):
   ```

3. **Batch Processing for Multiple Texts** (Optional)
   ```python
   texts = [text1, text2, text3]
   summaries = summarizer(texts, max_length=100, min_length=30, truncation=True)
   ```
---
## 🔢 Result

INPUT: 

Autonomous vehicles are transforming transportation by reducing human
intervention in driving. These vehicles use sensors, cameras, LiDAR,
and AI algorithms to navigate. The goal is to enhance road safety and
reduce traffic congestion. However, challenges such as
unpredictable human behavior, adverse weather conditions, and ethical
considerations in decision-making remain. Despite these hurdles,
continuous advancements in AI and machine learning are making self-driving 
technology more reliable and accessible.

OUTPUT:

Autonomous vehicles are transforming transportation by reducing human intervention in driving. 
These vehicles use sensors, cameras, LiDAR, and AI algorithms to navigate. The goal is to enhance 
road safety and reduce traffic congestion. But challenges such as unpredictable human behavior,
adverse weather conditions, and ethical considerations remain.
---
## 📜 License
This project is open-source and available under the **Apache 2.0 License**.

