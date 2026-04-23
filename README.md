# Retrieval-Augmented ABSA

Contrastive DeBERTa embedding + FAISS retrieval + multi-task BIO tagging and sentiment classification on SemEval 2015/2016 Restaurant.

## Setup

python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

## Tests

pytest tests/ -v
