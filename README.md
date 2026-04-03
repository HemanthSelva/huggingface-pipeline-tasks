# HuggingFace Pipeline Tasks

A complete implementation of 8 HuggingFace NLP Pipeline tasks using Transformers and Sentence-Transformers — built as part of my Data Science Internship at **Sourcesys Technologies**.

---

## Tasks Implemented

| # | Task | Model Used |
|---|------|-----------|
| 1 | Question Answering | `deepset/roberta-base-squad2` |
| 2 | Token Classification (NER) | `dbmdz/bert-large-cased-finetuned-conll03-english` |
| 3 | Text Classification (Sentiment Analysis) | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| 4 | Zero-Shot Classification | `facebook/bart-large-mnli` |
| 5 | Summarization | `facebook/bart-large-cnn` |
| 6 | Text Generation | `gpt2` |
| 7 | Sentence Similarity | `sentence-transformers/all-MiniLM-L6-v2` |
| 8 | Feature Extraction | `sentence-transformers/all-MiniLM-L6-v2` |

---

## How to Run

### Option 1 — Google Colab (Recommended)
1. Open [colab.research.google.com](https://colab.research.google.com)
2. File → Open notebook → GitHub → paste this repo URL
3. Open `HuggingFace_Pipeline_Tasks.ipynb`
4. Runtime → Run all

### Option 2 — Local
```bash
git clone https://github.com/HemanthSelva/huggingface-pipeline-tasks.git
cd huggingface-pipeline-tasks
pip install transformers torch sentence-transformers
jupyter notebook HuggingFace_Pipeline_Tasks.ipynb
```

---

## Dependencies

```
transformers
torch
sentence-transformers
numpy
```

---

## Repository Structure

```
huggingface-pipeline-tasks/
│
├── HuggingFace_Pipeline_Tasks.ipynb   # Main notebook with all 8 tasks
└── README.md                          # Project documentation
```

---

## Author

**HEMANTHSELVA A K**  
B.E. Computer Science & Engineering (Data Science)  
Bannari Amman Institute of Technology  
Data Science Intern — Sourcesys Technologies  
GitHub: [@HemanthSelva](https://github.com/HemanthSelva)
