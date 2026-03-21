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
LASSO_SRC    = os.path.join(REPO_ROOT, "Lasso", "src")
PROC_DIR     = os.path.join(REPO_ROOT, "Lasso", "data", "processed")
MODELS_PATH  = os.path.join(REPO_ROOT, "results", "models", "final_models.pkl")
METRICS_PATH = os.path.join(REPO_ROOT, "results", "metrics", "lasso_metrics.csv")
DB_PATH      = os.path.join(REPO_ROOT, "data", "sensory_database.json")

# Canonical sensory order – must match preprocess.py SENSORY_KEYS and plot_config.py SENSORY_ORDER
FEATURE_KEYS  = ["sweet", "bitter", "salty", "umami", "sour"]
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


# ── Sensory database (JSON) ───────────────────────────────────────────────────

def _load_db() -> list[dict]:
    with open(DB_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_db(entries: list[dict]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def _append_to_db(entry: dict) -> None:
    """Append a new researched ingredient to the JSON database."""
    entries = _load_db()
    # Guard against duplicates
    name_key = entry["name"].lower().strip()
    if any(e["name"].lower().strip() == name_key for e in entries):
        return
    entries.append(entry)
    _save_db(entries)


# Load DB names once at startup for matching
_db_entries: list[dict] = _load_db()
_db_names:   list[str]  = [e["name"] for e in _db_entries]
_db_map:     dict       = {e["name"]: e for e in _db_entries}


def _reload_db_cache() -> None:
    """Refresh in-memory DB cache after appending a new entry."""
    global _db_entries, _db_names, _db_map
    _db_entries = _load_db()
    _db_names   = [e["name"] for e in _db_entries]
    _db_map     = {e["name"]: e for e in _db_entries}


# ── Ingredient lookup ─────────────────────────────────────────────────────────

def _match_to_database(ingredient_names: list[str]) -> dict:
    """
    Batch-match ingredient names to the JSON database via Claude.
    Returns {ingredient_name: matched_db_name | null}.
    """
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


def _research_ingredient(name: str) -> dict:
    """
    Use Claude + web_search to find sensory scores for an unmatched ingredient.

    Always returns a dict with scores and one of three confidence levels:
      High   – solid cited scientific evidence
      Medium – scattered sources, reasonable estimate
      Low    – little/no sources, predicted from model knowledge
    """
    client = anthropic.Anthropic()
    prompt = (
        f'Research the sensory taste profile of "{name}" for use in food science.\n\n'
        "Find taste intensities on a 0–100 scale (Spectrum™ or equivalent) for:\n"
        "sweet, sour, bitter, umami, salty.\n\n"
        "Assign confidence based on what you find:\n"
        "HIGH   — solid cited scientific evidence (food science papers, official food databases)\n"
        "MEDIUM — scattered or indirect sources, enough to make a reasonable estimate\n"
        "LOW    — little to nothing found; predict from your food science knowledge\n\n"
        "You MUST always return scores. Never refuse or return null scores.\n\n"
        "Return ONLY this JSON (no markdown, no extra text):\n"
        '{\n'
        '  "sweet": <0-100>,\n'
        '  "sour": <0-100>,\n'
        '  "bitter": <0-100>,\n'
        '  "umami": <0-100>,\n'
        '  "salty": <0-100>,\n'
        '  "confidence": "High" | "Medium" | "Low",\n'
        '  "evidence": "<1-2 sentences citing sources or explaining reasoning>"\n'
        '}'
    )
    messages = [{"role": "user", "content": prompt}]

    def _parse_result(text: str) -> dict | None:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            result = json.loads(text)
            if all(k in result for k in ("sweet", "sour", "bitter", "umami", "salty", "confidence")):
                return result
        except Exception:
            pass
        return None

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
                        result = _parse_result(block.text)
                        if result:
                            return result
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
        print(f"[webapp] web research for '{name}' failed: {exc}")

    # Fallback: ask Claude without web search (always succeeds)
    try:
        fallback = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": (
                f'Predict the sensory taste profile of "{name}" from your food science knowledge.\n'
                'Return ONLY JSON: {"sweet":<0-100>,"sour":<0-100>,"bitter":<0-100>,'
                '"umami":<0-100>,"salty":<0-100>,"confidence":"Low",'
                '"evidence":"Predicted from model knowledge, no reliable sources found"}'
            )}],
        )
        result = _parse_result(fallback.content[0].text)
        if result:
            return result
    except Exception as exc:
        print(f"[webapp] fallback prediction for '{name}' failed: {exc}")

    # Hard fallback – neutral scores
    return {
        "sweet": 10.0, "sour": 10.0, "bitter": 10.0, "umami": 10.0, "salty": 10.0,
        "confidence": "Low",
        "evidence": "Predicted from model knowledge, no reliable sources found",
    }


# ── Ingredient enrichment ─────────────────────────────────────────────────────

def enrich_ingredients(ingredients: list) -> list:
    """
    Enrich each ingredient with sensory scores.

    Step 1 – Check JSON database (semantic match via Claude).
    Step 2 – Research with web search → High / Medium / Low confidence.
    Step 3 – Append researched ingredient to database for future lookups.

    Every ingredient always gets scores. Nothing is excluded.
    """
    names = [ing["name"] for ing in ingredients]

    # Step 1: batch-match all names against the database
    try:
        matches = _match_to_database(names)
    except Exception as exc:
        print(f"[webapp] DB matching failed: {exc}")
        matches = {}

    enriched = []
    for ing in ingredients:
        name     = ing["name"]
        db_match = matches.get(name)

        # ── database hit ──────────────────────────────────────────────────────
        if db_match and db_match in _db_map:
            entry = _db_map[db_match]
            scores = {k: entry[k] for k in ("sweet", "sour", "bitter", "umami", "salty")}
            enriched.append({
                **ing,
                "sensory_scores": scores,
                "source":         "database",
                "matched_name":   db_match,
                "confidence":     None,
                "evidence":       None,
            })
            continue

        # ── Step 2: research ──────────────────────────────────────────────────
        print(f"[webapp] Researching '{name}'…")
        result = _research_ingredient(name)

        scores = {k: float(result[k]) for k in ("sweet", "sour", "bitter", "umami", "salty")}
        confidence = result.get("confidence", "Low")
        evidence   = result.get("evidence", "")

        # Step 3: persist to database
        new_entry = {
            "name":       name,
            "sweet":      scores["sweet"],
            "sour":       scores["sour"],
            "bitter":     scores["bitter"],
            "umami":      scores["umami"],
            "salty":      scores["salty"],
            "predicted":  True,
            "confidence": confidence,
            "evidence":   evidence,
        }
        _append_to_db(new_entry)
        _reload_db_cache()

        enriched.append({
            **ing,
            "sensory_scores": scores,
            "source":         "predicted",
            "confidence":     confidence,
            "evidence":       evidence,
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


def _scrape_image_from_url(url: str) -> str | None:
    """Extract the best image from a recipe page (og:image → twitter:image → first large img)."""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FlavorLab/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 1. OpenGraph image (most reliable on recipe sites)
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            return og["content"]

        # 2. Twitter card image
        tw = soup.find("meta", attrs={"name": "twitter:image"})
        if tw and tw.get("content"):
            return tw["content"]

        # 3. First <img> with a src that looks like a photo (not icon/logo)
        for img in soup.find_all("img", src=True):
            src = img["src"]
            if any(src.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                if not any(skip in src.lower() for skip in ("icon", "logo", "avatar", "pixel", "1x1")):
                    # Make absolute if relative
                    if src.startswith("//"):
                        src = "https:" + src
                    elif src.startswith("/"):
                        from urllib.parse import urlparse
                        p = urlparse(url)
                        src = f"{p.scheme}://{p.netloc}{src}"
                    return src
    except Exception as exc:
        print(f"[webapp] image scrape failed for {url}: {exc}")
    return None


def _search_dish_image(recipe_name: str) -> str | None:
    """Use Claude + web_search to find a dish image URL."""
    client = anthropic.Anthropic()
    prompt = (
        f'Search the web for a photo of the dish "{recipe_name}". '
        'Find a direct image URL ending in .jpg, .png, or .webp from a reputable food or recipe site. '
        'Return ONLY valid JSON: {"image_url": "<URL or null>"}'
    )
    messages = [{"role": "user", "content": prompt}]

    def _parse(text: str) -> str | None:
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                if p.startswith("json"): p = p[4:]
                if "{" in p: text = p.strip(); break
        m = re.search(r'\{[^{}]*"image_url"\s*:\s*"([^"]+)"[^{}]*\}', text)
        if m:
            val = m.group(1)
            return val if val.lower() != "null" else None
        return None

    try:
        for _ in range(5):
            response = client.beta.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=messages,
                betas=["web-search-2025-03-05"],
            )
            if response.stop_reason == "end_turn":
                for block in reversed(response.content):
                    if hasattr(block, "text") and block.text.strip():
                        result = _parse(block.text)
                        if result:
                            return result
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
        print(f"[webapp] dish image search failed: {exc}")
    return None


def enrich_with_dish_info(recipe_name: str, source_url: str | None = None) -> dict:
    """
    Get dish image + description.

    If source_url is provided: scrape image directly from that page.
    Otherwise: use Claude web_search to find a photo of the dish.
    Description is always generated by Claude (no web search needed).
    """
    # ── Image ──────────────────────────────────────────────────────────────
    if source_url:
        image_url = _scrape_image_from_url(source_url)
        if not image_url:
            # Fallback: search anyway
            image_url = _search_dish_image(recipe_name)
    else:
        image_url = _search_dish_image(recipe_name)

    # ── Description (always from Claude, no web call needed) ───────────────
    try:
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": (
                f'Write a 2–3 sentence professional food science description of "{recipe_name}". '
                "Focus on its sensory character (taste, texture, aroma). Plain text only."
            )}],
        )
        description = msg.content[0].text.strip()
    except Exception as exc:
        print(f"[webapp] description generation failed: {exc}")
        description = (
            f"{recipe_name} is a prepared dish whose overall sensory profile emerges from "
            "the interaction of its ingredients across the five primary taste dimensions: "
            "sweet, bitter, salty, umami, and sour."
        )

    return {"image_url": image_url, "description": description}


def extract_recipe(text: str) -> dict:
    client  = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": f"{_EXTRACTION_PROMPT}\n\n{text}"}],
    )
    raw = message.content[0].text.strip()
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
        text       = ""
        source_url = None  # set only for URL input; used for image scraping

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
            source_url = request.form["url"].strip()
            try:
                text = fetch_url_text(source_url)
            except Exception as exc:
                return jsonify({"error": f"Could not fetch URL: {exc}"}), 400

        elif request.form.get("text", "").strip():
            text = request.form["text"].strip()

        else:
            return jsonify({"error": "No input provided"}), 400

        # 2. Claude extracts structured recipe (names + weights only)
        recipe = extract_recipe(text)

        # 3. Enrich every ingredient with sensory scores (no exclusions)
        recipe["ingredients"] = enrich_ingredients(recipe["ingredients"])

        # 4. Normalise weights and build feature vector from all ingredients
        all_norm = _normalize_weights(recipe["ingredients"])
        raw_vec  = _compute_feature_vector(all_norm)
        std_vec  = _standardize(raw_vec)
        x        = std_vec.reshape(1, -1)

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

        # 5. Store normalised weights back for display
        recipe["ingredients"] = all_norm

        dish_info = enrich_with_dish_info(recipe["recipe_name"], source_url=source_url)

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
