#!/usr/bin/env python3
"""
Flask web app for sensory score prediction using the Lasso model.
"""

import json
import re
from dotenv import load_dotenv
load_dotenv()
import os
import pickle
import subprocess
import sys
import tempfile

import anthropic
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LASSO_SRC    = os.path.join(REPO_ROOT, "Lasso_project", "src")
PROC_DIR     = os.path.join(REPO_ROOT, "Lasso_project", "data", "processed")
MODELS_PATH  = os.path.join(REPO_ROOT, "Lasso_project", "results", "models", "final_models.pkl")
METRICS_PATH = os.path.join(REPO_ROOT, "Lasso_project", "results", "metrics", "lasso_metrics.csv")
SCALER_MEAN  = os.path.join(PROC_DIR, "x_scaler_mean.npy")
SCALER_SCALE = os.path.join(PROC_DIR, "x_scaler_scale.npy")

# Order used by preprocess.py when building X feature vectors
FEATURE_KEYS = ["sweet", "bitter", "sour", "umami", "salty"]

# Sensory dimensions the models predict (keys in final_models.pkl)
SENSORY_ORDER = ["sweet", "bitter", "salty", "umami", "sour"]

# Make lasso.py importable (required to unpickle final_models.pkl)
sys.path.insert(0, LASSO_SRC)

app = Flask(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────

def _retrain() -> None:
    """Run preprocess.py then train.py to produce models."""
    env = os.environ.copy()
    env["PYTHONPATH"] = LASSO_SRC + os.pathsep + env.get("PYTHONPATH", "")
    for script in ("preprocess.py", "train.py"):
        subprocess.run(
            [sys.executable, os.path.join(LASSO_SRC, script)],
            cwd=LASSO_SRC,
            env=env,
            check=True,
        )


def load_models() -> dict:
    if not os.path.exists(MODELS_PATH):
        print("[webapp] Models not found – retraining…")
        _retrain()
    with open(MODELS_PATH, "rb") as fh:
        return pickle.load(fh)


def load_rmse() -> dict[str, float]:
    df = pd.read_csv(METRICS_PATH)
    return {
        row["target"]: float(row["RMSE"])
        for _, row in df.iterrows()
        if row["target"] in SENSORY_ORDER
    }


# Load once at startup
_models = load_models()
_rmse   = load_rmse()


# ── Feature engineering ───────────────────────────────────────────────────────

def _normalize_weights(ingredients: list) -> list:
    total = sum(float(i["weight"]) for i in ingredients)
    if total <= 0:
        total = 1.0
    return [
        {**i, "weight": float(i["weight"]) / total}
        for i in ingredients
    ]


def _compute_feature_vector(ingredients: list) -> np.ndarray:
    """Weighted average of sensory scores – order matches preprocess.py SENSORY_KEYS."""
    vec = np.zeros(len(FEATURE_KEYS), dtype=float)
    for ing in ingredients:
        w = float(ing["weight"])
        s = ing["sensory_scores"]
        vec += w * np.array([float(s.get(k, 0)) for k in FEATURE_KEYS], dtype=float)
    return vec


def _build_scaler_from_training_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Recompute the raw-feature scaler from training recipes.

    train.py overwrites the scaler files saved by preprocess.py with a
    second near-identity standardisation, so we cannot trust those files.
    We re-derive the original mean/std by replaying preprocess.py's logic
    on the raw_recipes data.
    """
    import importlib.util
    raw_path = os.path.join(REPO_ROOT, "data", "raw_recipes.py")
    spec = importlib.util.spec_from_file_location("raw_recipes_module", raw_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    recipes = mod.raw_recipes

    rows = []
    for r in recipes:
        ings = _normalize_weights(r["ingredients"])
        rows.append(_compute_feature_vector(ings))
    X_raw = np.vstack(rows)

    mean  = X_raw.mean(axis=0)
    std   = X_raw.std(axis=0)
    scale = np.where(std == 0, 1.0, std)
    return mean, scale


# Compute scaler once at module load
_scaler_mean, _scaler_scale = _build_scaler_from_training_data()


def _standardize(vec: np.ndarray) -> np.ndarray:
    return (vec - _scaler_mean) / _scaler_scale


# ── Claude extraction ─────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """
You are a data-extraction assistant. Your ONLY job is to parse the recipe text below
into structured JSON. Do NOT predict, estimate, or infer any sensory scores.

Return ONLY a valid JSON object with this exact structure (no markdown, no commentary):

{
  "recipe_name": "<string — infer a name if not stated>",
  "ingredients": [
    {
      "name": "<string>",
      "weight": <float 0-1, or null if not stated>,
      "sensory_scores": {
        "sweet":  <integer 0-100 ONLY if explicitly stated in the text, otherwise null>,
        "sour":   <integer 0-100 ONLY if explicitly stated in the text, otherwise null>,
        "bitter": <integer 0-100 ONLY if explicitly stated in the text, otherwise null>,
        "umami":  <integer 0-100 ONLY if explicitly stated in the text, otherwise null>,
        "salty":  <integer 0-100 ONLY if explicitly stated in the text, otherwise null>
      }
    }
  ]
}

Rules:
- Weights: normalise grams/percentages so they sum to 1.0. If no weights given, distribute evenly.
- Sensory scores: copy EXACTLY from the text if present. If a score is not stated, use null — never guess.
- Return ONLY the JSON object.

Recipe text:
""".strip()


def fetch_url_text(url: str) -> str:
    """Fetch a URL and return its readable text content."""
    import requests
    from bs4 import BeautifulSoup
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FlavorLab/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def enrich_with_dish_info(recipe_name: str) -> dict:
    """Find a dish image URL and description using Claude with web_search."""
    client = anthropic.Anthropic()
    prompt = (
        f'Search the web for "{recipe_name}" dish. '
        'Find a direct image URL (ending in .jpg, .png, or .webp from a public food/recipe site). '
        'Also write a 2–3 sentence professional food science description of the dish. '
        'Return ONLY valid JSON: {"image_url": "<URL or null>", "description": "<text>"}'
    )
    messages = [{"role": "user", "content": prompt}]

    def _parse(text: str) -> dict | None:
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                if p.startswith("json"): p = p[4:]
                if "{" in p:
                    text = p.strip(); break
        m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except Exception: pass
        return None

    try:
        for _ in range(5):
            response = client.beta.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=messages,
                betas=["web-search-2025-03-05"],
            )
            if response.stop_reason == "end_turn":
                for block in reversed(response.content):
                    if hasattr(block, "text") and block.text.strip():
                        result = _parse(block.text)
                        if result: return result
                break
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": ""}
                    for b in response.content if b.type == "tool_use"
                ]})
            else:
                break
    except Exception as exc:
        print(f"[webapp] dish info enrichment failed: {exc}")

    return {
        "image_url": None,
        "description": (
            f"{recipe_name} is a prepared dish whose overall sensory profile emerges from "
            "the interaction of its ingredients across the five primary taste dimensions: "
            "sweet, bitter, salty, umami, and sour."
        ),
    }


def extract_recipe(text: str) -> dict:
    client  = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": f"{_EXTRACTION_PROMPT}\n\n{text}"}],
    )
    raw = message.content[0].text.strip()
    # Strip any accidental code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Collect text from upload or form field
        text = ""
        if "file" in request.files and request.files["file"].filename:
            file     = request.files["file"]
            filename = file.filename.lower()

            if filename.endswith(".txt"):
                text = file.read().decode("utf-8", errors="replace")

            elif filename.endswith(".pdf"):
                import fitz  # PyMuPDF
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    file.save(tmp.name)
                    tmp_path = tmp.name
                doc  = fitz.open(tmp_path)
                text = " ".join(page.get_text() for page in doc)
                doc.close()
                os.unlink(tmp_path)

            elif filename.endswith(".docx"):
                from docx import Document
                with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                    file.save(tmp.name)
                    tmp_path = tmp.name
                doc  = Document(tmp_path)
                text = " ".join(p.text for p in doc.paragraphs)
                os.unlink(tmp_path)

            else:
                return jsonify({"error": "Unsupported file type. Use .txt, .pdf, or .docx"}), 400

        elif request.form.get("url", "").strip():
            try:
                text = fetch_url_text(request.form["url"].strip())
            except Exception as exc:
                return jsonify({"error": f"Could not fetch URL: {exc}"}), 400

        elif request.form.get("text", "").strip():
            text = request.form["text"].strip()

        else:
            return jsonify({"error": "No input provided"}), 400

        # 2. Claude extracts structured recipe
        recipe = extract_recipe(text)

        # 3. Validate all sensory scores are present
        missing = []
        for ing in recipe["ingredients"]:
            s = ing.get("sensory_scores", {})
            null_keys = [k for k in FEATURE_KEYS if s.get(k) is None]
            if null_keys:
                missing.append(f"{ing['name']} (missing: {', '.join(null_keys)})")
        if missing:
            return jsonify({
                "error": (
                    "Sensory scores are required for every ingredient but were not found in your input. "
                    "Please provide scores (0–100) for: " + "; ".join(missing)
                ),
                "missing_scores": True,
                "ingredients": recipe["ingredients"],
            }), 422

        # 4. Normalise weights
        recipe["ingredients"] = _normalize_weights(recipe["ingredients"])

        # 4. Feature vector → standardise → predict
        raw_vec = _compute_feature_vector(recipe["ingredients"])
        std_vec = _standardize(raw_vec)
        x       = std_vec.reshape(1, -1)

        predictions = {}
        confidence  = {}
        for taste in SENSORY_ORDER:
            pred  = float(_models[taste].predict(x)[0])
            pred  = max(0.0, min(100.0, pred))
            rmse  = _rmse.get(taste, 0.0)
            predictions[taste] = round(pred, 1)
            confidence[taste]  = {
                "lower": round(max(0.0, pred - rmse), 1),
                "upper": round(min(100.0, pred + rmse), 1),
            }

        dish_info = enrich_with_dish_info(recipe["recipe_name"])

        return jsonify({
            "recipe":      recipe,
            "predictions": predictions,
            "confidence":  confidence,
            "dish_info":   dish_info,
        })

    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Could not parse recipe JSON from Claude: {exc}"}), 422
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
