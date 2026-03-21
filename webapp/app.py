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
EXCEL_PATH   = os.path.join(REPO_ROOT, "data", "Supplementary_Data_File_1_v10.xlsx")
PREDICTED_DATA_PATH = os.path.join(REPO_ROOT, "data", "predicted_sensory_data.py")

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


# ── Sensory database (Excel) ──────────────────────────────────────────────────

def _load_sensory_database() -> tuple[list, dict]:
    """Load the 'foods' sheet and return (list_of_names, name→scores dict)."""
    import openpyxl
    wb   = openpyxl.load_workbook(EXCEL_PATH, read_only=True, data_only=True)
    ws   = wb["foods"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header = rows[0]
    col    = {h: i for i, h in enumerate(header) if h is not None}

    def safe(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    db_names = []
    db_map   = {}
    for row in rows[1:]:
        name = row[col.get("Food Name (EN)", 1)]
        if not name:
            continue
        scores = {
            "sweet":  safe(row[col.get("Sweet Mean",  4)]),
            "sour":   safe(row[col.get("Sour Mean",   7)]),
            "bitter": safe(row[col.get("Bitter Mean", 10)]),
            "umami":  safe(row[col.get("Umami Mean",  13)]),
            "salty":  safe(row[col.get("Salt Mean",   16)]),
        }
        if any(v is None for v in scores.values()):
            continue
        name_str = str(name)
        db_names.append(name_str)
        db_map[name_str] = scores
    return db_names, db_map


_db_names, _db_map = _load_sensory_database()


# ── Predicted sensory data cache ──────────────────────────────────────────────

def _load_predicted() -> list:
    if not os.path.exists(PREDICTED_DATA_PATH):
        return []
    import importlib.util
    spec = importlib.util.spec_from_file_location("predicted_sensory_data", PREDICTED_DATA_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(getattr(mod, "predicted_ingredients", []))


def _get_cached(name: str) -> dict | None:
    name_key = name.lower().strip()
    for entry in _load_predicted():
        if entry.get("name", "").lower().strip() == name_key:
            return entry
    return None


def _save_predicted(name: str, result: dict) -> None:
    predicted = _load_predicted()
    name_key  = name.lower().strip()
    if any(e.get("name", "").lower().strip() == name_key for e in predicted):
        return  # already cached
    predicted.append({
        "name":           name,
        "sensory_scores": result["sensory_scores"],
        "confidence":     result.get("confidence", 0.7),
        "evidence":       result.get("evidence", ""),
        "source_url":     result.get("source_url", None),
    })
    lines = [
        "# Auto-generated by FlavorLab webapp — do not edit manually\n",
        "# Structure: name, sensory_scores (0-100), confidence (0-1), evidence, source_url\n",
        "predicted_ingredients = [\n",
    ]
    for e in predicted:
        lines.append(f"    {repr(e)},\n")
    lines.append("]\n")
    with open(PREDICTED_DATA_PATH, "w") as fh:
        fh.writelines(lines)


# ── Ingredient enrichment ─────────────────────────────────────────────────────

def _match_to_database(ingredient_names: list[str]) -> dict:
    """Batch-match ingredient names to the Excel database via Claude."""
    db_list_str = "\n".join(f"- {n}" for n in _db_names)
    prompt = (
        "Match each ingredient to the closest entry in the food database below.\n"
        "Return ONLY a JSON object mapping each ingredient name to the best matching "
        "database entry name, or null if there is no reasonable match.\n\n"
        "Rules:\n"
        "- Same ingredient = match (e.g. 'apple' → 'Apple, raw with skin')\n"
        "- Clearly different ingredient = null (e.g. 'salt' vs 'MSG' = null)\n"
        "- Favour specificity: 'whole milk' → closest milk variant in the list\n\n"
        f"Ingredients to match:\n{json.dumps(ingredient_names)}\n\n"
        f"Available database entries:\n{db_list_str}\n\n"
        f'Return ONLY valid JSON like: {{"apple": "Apple, raw with skin", "unicorn dust": null}}'
    )
    message = anthropic.Anthropic().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _research_ingredient(name: str) -> dict | None:
    """Use Claude + web_search to find sensory scores for an unmatched ingredient."""
    client  = anthropic.Anthropic()
    prompt  = (
        f'Search for scientific evidence about the sensory taste profile of "{name}".\n\n'
        "Find peer-reviewed research or validated food science databases reporting taste "
        "intensities for: sweet, sour, bitter, umami, and salty on a 0–100 scale "
        "(Spectrum™ scale or equivalent).\n\n"
        "If you find reliable evidence, return ONLY this JSON:\n"
        '{"sensory_scores": {"sweet": <0-100>, "sour": <0-100>, "bitter": <0-100>, '
        '"umami": <0-100>, "salty": <0-100>}, "confidence": <0.0-1.0>, '
        '"evidence": "<1-2 sentence summary of source>", "source_url": "<URL or null>"}\n\n'
        'If you cannot find reliable scientific evidence, return: {"confident": false}\n\n'
        "Return ONLY valid JSON."
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        for _ in range(6):
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
                        raw = block.text.strip()
                        if raw.startswith("```"):
                            raw = raw.split("```")[1]
                            if raw.lower().startswith("json"):
                                raw = raw[4:]
                            raw = raw.strip()
                        try:
                            result = json.loads(raw)
                            if result.get("confident") is False:
                                return None
                            if "sensory_scores" in result:
                                return result
                        except Exception:
                            pass
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
        print(f"[webapp] research for '{name}' failed: {exc}")
    return None


def enrich_ingredients(ingredients: list) -> list:
    """
    Enrich each ingredient with sensory scores from database, cache, or research.
    Tags each ingredient with source: 'database' | 'researched' | 'unknown'.
    """
    names = [ing["name"] for ing in ingredients]

    # Step 1: batch-match all names against the Excel database
    try:
        matches = _match_to_database(names)
    except Exception as exc:
        print(f"[webapp] DB matching failed: {exc}")
        matches = {}

    enriched = []
    for ing in ingredients:
        name     = ing["name"]
        db_match = matches.get(name)

        # ── database match ────────────────────────────────────────────────────
        if db_match and db_match in _db_map:
            enriched.append({
                **ing,
                "sensory_scores": _db_map[db_match],
                "source":         "database",
                "matched_name":   db_match,
            })
            continue

        # ── local cache (previously researched) ──────────────────────────────
        cached = _get_cached(name)
        if cached:
            enriched.append({
                **ing,
                "sensory_scores": cached["sensory_scores"],
                "source":         "researched",
                "confidence":     cached.get("confidence", 0.7),
                "evidence":       cached.get("evidence", ""),
                "source_url":     cached.get("source_url", None),
            })
            continue

        # ── web research ──────────────────────────────────────────────────────
        result = _research_ingredient(name)
        if result:
            _save_predicted(name, result)
            enriched.append({
                **ing,
                "sensory_scores": result["sensory_scores"],
                "source":         "researched",
                "confidence":     result.get("confidence", 0.7),
                "evidence":       result.get("evidence", ""),
                "source_url":     result.get("source_url", None),
            })
        else:
            enriched.append({
                **ing,
                "sensory_scores": {"sweet": 0, "sour": 0, "bitter": 0, "umami": 0, "salty": 0},
                "source":         "unknown",
            })

    return enriched


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
into structured JSON with ingredient names and weights.
Do NOT include or predict any sensory scores.

Return ONLY a valid JSON object with this exact structure (no markdown, no commentary):

{
  "recipe_name": "<string — infer a name if not stated>",
  "ingredients": [
    {
      "name": "<string>",
      "weight": <float 0-1>
    }
  ]
}

Rules:
- Weights: normalise grams/percentages so they sum to 1.0. If no weights given, distribute evenly.
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

        # 2. Claude extracts structured recipe (names + weights only)
        recipe = extract_recipe(text)

        # 3. Enrich each ingredient with sensory scores from database / research
        recipe["ingredients"] = enrich_ingredients(recipe["ingredients"])

        # 4. Use only known ingredients for the Lasso feature vector
        known = [i for i in recipe["ingredients"] if i["source"] != "unknown"]
        if not known:
            return jsonify({"error": "No sensory data could be found for any ingredient."}), 422

        known_norm = _normalize_weights(known)
        raw_vec    = _compute_feature_vector(known_norm)
        std_vec    = _standardize(raw_vec)
        x          = std_vec.reshape(1, -1)

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

        # 5. Normalise all weights for display (including unknowns)
        recipe["ingredients"] = _normalize_weights(recipe["ingredients"])

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
