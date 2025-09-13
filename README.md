# Sitemind

<img width="1023" height="605" alt="Screenshot 2025-09-13 at 1 51 40 PM" src="https://github.com/user-attachments/assets/7dd78ae9-1a61-424e-a9d6-d5827ecb8303" />


Local platform for ingesting a website, building a knowledge base (FAISS or NumPy fallback), generating strategies (rule-based for MVP), chatting over the KB, and exporting results.

## Quickstart

1. Create venv

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install deps (Poetry or pip)

```bash
# with poetry
pip install poetry
poetry install

# or with pip
pip install -r requirements.txt
```

3. Run the Streamlit app


```bash
streamlit run app/main.py
```

## Notes
- Everything runs locally by default, no paid APIs required.
- First run will download the SentenceTransformer model (~90MB).
- If FAISS install fails, the app falls back to a NumPy-based similarity search.
