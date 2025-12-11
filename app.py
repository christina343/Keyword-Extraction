"""Flask/FastAPI powered keyword extraction site."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from starlette.applications import Starlette
from starlette.routing import Mount


def simple_normalize(token: str) -> str:
    """Apply a lightweight stemming-like normalization."""
    endings = ("ingly", "edly", "ing", "ed", "ly", "es", "s")
    for ending in endings:
        if token.endswith(ending) and len(token) - len(ending) >= 3:
            return token[: -len(ending)]
    return token


def preprocess_text(text: str, normalize: bool) -> str:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if normalize:
        tokens = [simple_normalize(tok) for tok in tokens]
    return " ".join(tokens)


def extract_keywords(
    text: str,
    ratio: float = 0.2,
    max_keywords: int | None = 20,
    use_lemmatization: bool = True,
) -> List[str]:
    """Extract keywords by ranking TF-IDF scores."""
    clean_text = (text or "").strip()
    if not clean_text:
        return []

    processed = preprocess_text(clean_text, normalize=use_lemmatization)
    if not processed:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([processed])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()

    ranked = sorted(zip(scores, feature_names), key=lambda item: item[0], reverse=True)
    if not ranked:
        return []

    ratio_limit = max(1, int(len(ranked) * ratio))
    hard_limit = max_keywords or len(ranked)
    final_limit = max(1, min(ratio_limit, hard_limit))

    return [term for _, term in ranked[:final_limit]]


@dataclass
class Insights:
    words: int
    characters: int
    sentences: int


def compute_insights(text: str) -> Insights:
    if not text:
        return Insights(words=0, characters=0, sentences=0)
    words = len(text.split())
    chars = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return Insights(words=words, characters=chars, sentences=max(1, sentences))


def load_text_from_file(upload) -> str:
    if upload is None or upload.filename == "":
        return ""
    suffix = Path(upload.filename).suffix.lower()
    raw = upload.read()
    if suffix in {".txt", ".md"}:
        return raw.decode("utf-8", errors="ignore")
    raise ValueError("Unsupported file. Please upload .txt or .md.")


def create_flask_app() -> Flask:
    flask_app = Flask(__name__, template_folder="templates", static_folder="static")
    flask_app.secret_key = "keyword-lab"

    sample_text = (
        "In a bustling city where information overflowed like a torrent, Emma, an "
        "aspiring journalist, struggled to uncover the truth amidst a sea of noise."
        " Assigned to investigate a high-profile scandal, she turned to data-driven "
        "methods to separate fact from fiction."
    )

    @flask_app.route("/", methods=["GET", "POST"])
    def index():
        text_input = sample_text
        ratio = 0.35
        max_keywords = 25
        normalize = True
        keywords: Sequence[str] = []
        error: str | None = None

        if request.method == "POST":
            ratio = float(request.form.get("ratio", ratio))
            max_keywords = int(request.form.get("max_keywords", max_keywords))
            normalize = bool(request.form.get("normalize"))
            text_input = request.form.get("text_input", "")

            upload = request.files.get("document")
            if upload and upload.filename:
                try:
                    text_input = load_text_from_file(upload)
                except ValueError as exc:
                    error = str(exc)
                    flash(error, "warning")

            keywords = extract_keywords(
                text_input,
                ratio=ratio,
                max_keywords=max_keywords,
                use_lemmatization=normalize,
            )

        insights = compute_insights(text_input)
        keywords_payload = "\n".join(keywords)

        return render_template(
            "index.html",
            text_input=text_input,
            ratio=ratio,
            max_keywords=max_keywords,
            normalize=normalize,
            keywords=keywords,
            insights=insights,
            keywords_payload=keywords_payload,
        )

    @flask_app.post("/download")
    def download_keywords():
        payload = request.form.get("keywords_payload", "")
        if not payload.strip():
            flash("Nothing to download yet.", "warning")
            return redirect(url_for("index"))
        buffer = io.BytesIO(payload.encode("utf-8"))
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype="text/plain",
            as_attachment=True,
            download_name="keywords.txt",
        )

    return flask_app


class KeywordRequest(BaseModel):
    text: str = Field("", description="Text to analyze.")
    ratio: float = Field(0.2, ge=0.05, le=0.9)
    max_keywords: int = Field(20, ge=1, le=100)
    normalize: bool = Field(True, description="Apply normalization.")


def create_fastapi_app() -> FastAPI:
    api = FastAPI(
        title="Keyword Extraction Lab API",
        version="2.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    @api.post("/keywords")
    async def api_keywords(payload: KeywordRequest) -> JSONResponse:
        keywords = extract_keywords(
            payload.text,
            ratio=payload.ratio,
            max_keywords=payload.max_keywords,
            use_lemmatization=payload.normalize,
        )
        return JSONResponse({"keywords": keywords, "count": len(keywords)})

    return api


def build_application() -> Starlette:
    api_app = create_fastapi_app()
    flask_app = create_flask_app()
    starlette_app = Starlette()
    starlette_app.mount("/api", api_app)
    starlette_app.mount("/", WSGIMiddleware(flask_app))
    return starlette_app


app = build_application()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
