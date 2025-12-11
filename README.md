# Keyword Extraction 

A Flask + FastAPI web application that uses TF-IDF ranking (via scikit-learn) to extract the most relevant keywords from any passage of text. Built to assist investigative journalists and researchers who need to sift quickly through large corpora and surface the essential concepts.

## Features
- Paste or type any text, or upload a `.txt/.md` document, then extract keywords with one click.
- Tune keyword density, maximum keyword count, and an optional normalization toggle.
- Responsive HTML/CSS interface served through Flask, plus a `/api/keywords` FastAPI endpoint for headless use.

## Requirements
- Python 3.10+
- See `requirements.txt` for Python dependencies (`flask`, `fastapi`, `uvicorn`, `scikit-learn`, etc.).

## Local Development / Hosting
1. **Clone or download** this project into your workspace.
2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. **Run the ASGI server**:
   ```bash
   uvicorn app:app --reload
   ```
5. **Open the browser** at `http://localhost:8000/` for the Flask UI. The API lives under `http://localhost:8000/api/keywords`.

### Deploying
- **Uvicorn/Gunicorn**: run `uvicorn app:app --host 0.0.0.0 --port $PORT`.
- **Render / Railway / Fly.io / etc.**: configure a web service with start command `uvicorn app:app --host 0.0.0.0 --port $PORT`.
- **Docker (optional)**: base on `python:3.11-slim`, install requirements, expose port, and run the uvicorn command.

## Testing & Validation
- The logic for keyword extraction lives in `extract_keywords`. You can validate it with `pytest` (after installing `pytest`) using the tests in `tests/`.
- Manual validation: run the app, paste both short and long texts, upload `.txt` files, and confirm that adjusting sliders changes the keyword output as expected.

## Project Structure
```
app.py             # Shared logic + FastAPI API + mounted Flask UI
templates/         # HTML files for the Flask interface
static/            # CSS assets
README.md          # Documentation & hosting instructions
requirements.txt   # Dependencies
tests/             # Pytest smoke tests
```
