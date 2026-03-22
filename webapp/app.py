#!/usr/bin/env python3
"""
Flask web app for sensory score prediction using the Lasso model.
"""

import json
import re
import uuid as _uuid
from dotenv import load_dotenv
load_dotenv()
import os
import pickle
import sys
import tempfile

import anthropic
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ── Paths ────────────────────────────────────────────────────────────────────
WEBAPP_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.abspath(os.path.join(WEBAPP_DIR, ".."))
MODELS_DIR   = os.path.join(REPO_ROOT, "results", "models")
MODELS_PATH  = os.path.join(MODELS_DIR, "final_models.pkl")
METRICS_PATH = os.path.join(REPO_ROOT, "results", "metrics", "lasso_metrics.csv")
EXCEL_PATH   = os.path.join(REPO_ROOT, "data", "Supplementary_Data_File_1_v11.xlsx")
EXCEL_SHEET  = "foods"
SCALER_MEAN  = os.path.join(MODELS_DIR, "x_scaler_mean.npy")
SCALER_SCALE = os.path.join(MODELS_DIR, "x_scaler_scale.npy")

# Excel column → internal key mapping
COL_NAME    = "Food Name (EN)"
COL_SWEET   = "Sweet Mean"
COL_SOUR    = "Sour Mean"
COL_BITTER  = "Bitter Mean"
COL_UMAMI   = "Umami Mean"
COL_SALTY   = "Salt Mean"
COL_PRED    = "predicted"
COL_CONF    = "confidence"
COL_EVID    = "evidence"

# Canonical sensory order
FEATURE_KEYS  = ["sweet", "bitter", "salty", "umami", "sour"]
SENSORY_ORDER = ["sweet", "bitter", "salty", "umami", "sour"]

# Make lasso.py importable (required to unpickle final_models.pkl)
sys.path.insert(0, WEBAPP_DIR)

app = Flask(__name__)

# ── In-memory session store ────────────────────────────────────────────────────
_sessions: dict = {}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models() -> dict:
    if not os.path.exists(MODELS_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODELS_PATH}\n"
            "Run the training pipeline on the main branch to generate models."
        )
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


# ── Sensory database (Excel – foods sheet) ────────────────────────────────────

def _load_db_df() -> pd.DataFrame:
    return pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET, engine="openpyxl")


def _append_to_db(new_row: dict) -> None:
    """Append a researched ingredient row and save back, preserving all other sheets."""
    df = _load_db_df()
    # Guard against duplicates
    name_key = str(new_row[COL_NAME]).lower().strip()
    if any(str(v).lower().strip() == name_key for v in df[COL_NAME].dropna()):
        return
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=EXCEL_SHEET, index=False)


def _build_cache(df: pd.DataFrame) -> tuple[list[str], dict]:
    names, mapping = [], {}
    for _, row in df.iterrows():
        name = row.get(COL_NAME)
        if not name or pd.isna(name):
            continue
        try:
            scores = {
                "sweet":  float(row[COL_SWEET]),
                "sour":   float(row[COL_SOUR]),
                "bitter": float(row[COL_BITTER]),
                "umami":  float(row[COL_UMAMI]),
                "salty":  float(row[COL_SALTY]),
            }
        except (KeyError, ValueError, TypeError):
            continue
        if any(pd.isna(v) for v in scores.values()):
            continue
        predicted = str(row.get(COL_PRED, "")).strip().lower() == "yes"
        mapping[str(name)] = {
            **scores,
            "source":     "predicted" if predicted else "database",
            "confidence": row.get(COL_CONF) if predicted else None,
            "evidence":   row.get(COL_EVID)  if predicted else None,
        }
        names.append(str(name))
    return names, mapping


# Load DB once at startup
_db_names: list[str] = []
_db_map:   dict      = {}
_db_names, _db_map = _build_cache(_load_db_df())


def _reload_db_cache() -> None:
    global _db_names, _db_map
    _db_names, _db_map = _build_cache(_load_db_df())


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
        "- Match ONLY to the ingredient in its pure/standalone form (e.g. 'apple' → 'Apple, raw with skin').\n"
        "- Return null if the best database entry is the ingredient combined with or served alongside "
        "another food (e.g. 'granulated sugar' must NOT match 'Sugar granulated, with coffee' or "
        "'Sugar granulated, with tea' — those entries reflect a coffee/tea context, not pure sugar).\n"
        "- Return null if the ingredient and database entry are clearly different foods "
        "(e.g. 'salt' vs 'MSG' = null; 'flour' vs 'Crisps based on potato flour' = null).\n"
        "- Favour the most specific *pure-ingredient* match: 'whole milk' → closest plain milk variant.\n"
        "- When in doubt, return null — it is better to research the ingredient than to use a wrong match.\n\n"
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
    Use Claude Opus + web_search to find sensory scores for an unmatched ingredient.
    Single call — Anthropic handles tool execution server-side.

    Always returns a dict with scores and one of three confidence levels:
      High   – direct measurement from a cited paper or database
      Medium – indirect/related sources; reasoned estimate
      Low    – no sources; predicted from food science knowledge
    """
    def _parse_result(text: str) -> dict | None:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != '{':
                continue
            try:
                obj, _ = decoder.raw_decode(text, i)
                if all(k in obj for k in ("sweet", "sour", "bitter", "umami", "salty", "confidence")):
                    return obj
            except json.JSONDecodeError:
                continue
        return None

    prompt = (
        f'Search for the sensory taste profile of "{name}" using scientific sources.\n\n'
        "Look for peer-reviewed food science papers, official food databases (Wageningen SVT, "
        "USDA, FlavorDB, Nizo), or validated sensory studies reporting Spectrum™ scale scores "
        "(0–100) for: sweet, sour, bitter, umami, salty.\n\n"
        "Confidence rules:\n"
        "High   — direct measurement from a cited paper or database (include DOI/URL)\n"
        "Medium — indirect or related sources; estimate with reasoning\n"
        "Low    — no sources found; predict from food science knowledge\n\n"
        "You MUST always return numeric scores. Never refuse.\n"
        "evidence: cite the actual source with author, year, DOI/URL if available.\n\n"
        "Return ONLY this JSON (no markdown, no extra text):\n"
        '{"sweet":<0-100>,"sour":<0-100>,"bitter":<0-100>,"umami":<0-100>,"salty":<0-100>,'
        '"confidence":"High"|"Medium"|"Low","evidence":"<cited source or reasoning>"}'
    )

    # Primary: Opus + web search
    print(f"[research] '{name}' — calling claude-opus-4-6 with web_search…")
    try:
        response = anthropic.Anthropic().beta.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
            betas=["web-search-2025-03-05"],
        )
        # Log every block so we can see what the model did
        for i, block in enumerate(response.content):
            btype = getattr(block, "type", type(block).__name__)
            if btype == "tool_use":
                query = getattr(block, "input", {}).get("query", "?")
                print(f"[research]   block[{i}] web_search query: {query!r}")
            elif btype == "tool_result":
                print(f"[research]   block[{i}] web_search result received")
            elif hasattr(block, "text") and block.text.strip():
                snippet = block.text.strip()[:120].replace("\n", " ")
                print(f"[research]   block[{i}] text: {snippet}")

        full_text = " ".join(
            block.text for block in response.content
            if hasattr(block, "text") and block.text.strip()
        )
        result = _parse_result(full_text)
        if result:
            print(f"[research] '{name}' — confidence={result.get('confidence')} evidence={str(result.get('evidence',''))[:80]}")
            return result
        print(f"[research] '{name}' — parse failed, raw text: {full_text[:200]}")
    except Exception as exc:
        print(f"[research] '{name}' — Opus+web_search FAILED: {exc}")

    # Fallback: Opus without web search
    print(f"[research] '{name}' — falling back to claude-opus-4-6 (no web search)…")
    try:
        response = anthropic.Anthropic().beta.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": (
                f'Predict the sensory taste profile of "{name}" from your food science knowledge.\n'
                'Return ONLY JSON: {"sweet":<0-100>,"sour":<0-100>,"bitter":<0-100>,'
                '"umami":<0-100>,"salty":<0-100>,"confidence":"Low",'
                '"evidence":"Predicted from model knowledge, no reliable sources found"}'
            )}],
        )
        result = _parse_result(response.content[0].text)
        if result:
            print(f"[research] '{name}' — fallback succeeded (confidence=Low)")
            return result
        print(f"[research] '{name}' — fallback parse also failed")
    except Exception as exc:
        print(f"[research] '{name}' — fallback FAILED: {exc}")

    print(f"[research] '{name}' — all strategies failed, using hardcoded defaults")
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
                "source":         entry["source"],
                "matched_name":   db_match,
                "confidence":     entry["confidence"],
                "evidence":       entry["evidence"],
            })
            continue

        # ── Step 2: research ──────────────────────────────────────────────────
        print(f"[webapp] Researching '{name}'…")
        result = _research_ingredient(name)

        scores = {k: float(result[k]) for k in ("sweet", "sour", "bitter", "umami", "salty")}
        confidence = result.get("confidence", "Low")
        evidence   = result.get("evidence", "")

        # Step 3: persist to Excel
        new_entry = {
            COL_NAME:   name,
            COL_SWEET:  scores["sweet"],
            COL_SOUR:   scores["sour"],
            COL_BITTER: scores["bitter"],
            COL_UMAMI:  scores["umami"],
            COL_SALTY:  scores["salty"],
            COL_PRED:   "Yes",
            COL_CONF:   confidence,
            COL_EVID:   evidence,
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


def _load_scaler() -> tuple[np.ndarray, np.ndarray]:
    """Load pre-computed feature scaler from results/models/."""
    if not os.path.exists(SCALER_MEAN) or not os.path.exists(SCALER_SCALE):
        raise FileNotFoundError(
            f"Scaler files not found in {MODELS_DIR}.\n"
            "Run the training pipeline on the main branch to generate them."
        )
    return np.load(SCALER_MEAN), np.load(SCALER_SCALE)


# Load scaler once at startup
_scaler_mean, _scaler_scale = _load_scaler()


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


def _find_dish_image(recipe_name: str) -> str | None:
    """
    Multi-strategy image search. Tries in order until one succeeds:
      1. TheMealDB  – free food API with photos, great for common dishes
      2. Wikipedia  – article thumbnail, broad coverage
      3. Wikimedia Commons – huge free image database
      4. Claude web_search – last resort, scans the web for a direct image URL
    """
    import requests
    from urllib.parse import quote_plus
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FlavorLab/1.0)"}

    # 1. TheMealDB (food-specific, free, no key)
    try:
        r = requests.get(
            f"https://www.themealdb.com/api/json/v1/1/search.php?s={quote_plus(recipe_name)}",
            headers=headers, timeout=8
        )
        if r.status_code == 200:
            meals = r.json().get("meals") or []
            if meals and meals[0].get("strMealThumb"):
                return meals[0]["strMealThumb"]
    except Exception:
        pass

    # 2. Wikipedia article thumbnail — try full name, 2-word prefix, then individual words
    _stop = {"with", "and", "the", "in", "on", "a", "of", "glazed", "roasted",
             "grilled", "baked", "fried", "steamed", "fresh", "dried"}
    _keywords = [w for w in recipe_name.split() if w.lower() not in _stop and len(w) > 3]
    candidates = [recipe_name, " ".join(recipe_name.split()[:2])] + _keywords
    for candidate in candidates:
        try:
            r = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(candidate)}",
                headers=headers, timeout=8
            )
            if r.status_code == 200:
                img = r.json().get("thumbnail", {}).get("source")
                if img:
                    return img
        except Exception:
            pass

    # 3. Wikimedia Commons image search
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query", "generator": "search", "gsrnamespace": "6",
                "gsrsearch": f"food {recipe_name}", "gsrlimit": "5",
                "prop": "imageinfo", "iiprop": "url", "iiurlwidth": "600",
                "format": "json",
            },
            headers=headers, timeout=8
        )
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                info = (page.get("imageinfo") or [{}])[0]
                url = info.get("url", "")
                if url and any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                    return url
    except Exception:
        pass

    # 4. Claude web_search – last resort (also catches CDN URLs without extensions)
    try:
        response = anthropic.Anthropic().beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": (
                f'Find a photo of "{recipe_name}" food dish. '
                f'Return ONLY a single direct image URL (must end in .jpg, .jpeg, .png, or .webp). '
                f'No text, no explanation, just the URL.'
            )}],
            betas=["web-search-2025-03-05"],
        )
        full_text = " ".join(
            block.text for block in response.content
            if hasattr(block, "text") and block.text.strip()
        ).strip()
        # Try strict image extension match first, then any URL from image CDNs
        m = (re.search(r'https?://\S+\.(?:jpg|jpeg|png|webp)(?:\?\S*)?', full_text, re.IGNORECASE)
             or re.search(r'https?://(?:upload\.wikimedia\.org|images\.\S+|img\.\S+|cdn\.\S+|photos\.\S+)/\S+', full_text, re.IGNORECASE))
        if m:
            return m.group().rstrip('.,;)"\'')
    except Exception as exc:
        print(f"[webapp] Claude image search failed for '{recipe_name}': {exc}")

    return None


def enrich_with_dish_info(recipe_name: str, source_url: str | None = None) -> dict:
    """
    Get dish image + description.
    If source_url provided: scrape og:image from that page, else search for the dish.
    """
    # ── Image ──────────────────────────────────────────────────────────────
    image_url = None
    if source_url:
        image_url = _scrape_image_from_url(source_url)
    if not image_url:
        image_url = _find_dish_image(recipe_name)

    # ── Description (always from Claude, no web call needed) ───────────────
    try:
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=160,
            messages=[{"role": "user", "content": (
                f'Write 2–3 sentences describing the sensory and flavour chemistry of "{recipe_name}" '
                f'for a food scientist. Cover dominant taste attributes (sweet/sour/bitter/umami/salty), '
                f'key flavour-active compounds or Maillard/fermentation reactions if relevant, '
                f'and any notable texture or aroma interactions. Plain text only, no bullet points.'
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


# ── Prediction helper (reused by /predict and /chat) ─────────────────────────

def _run_prediction_from_recipe(recipe: dict) -> tuple[dict, dict]:
    """Run Lasso prediction on an already-enriched recipe. Normalises weights in-place."""
    all_norm = _normalize_weights(recipe["ingredients"])
    raw_vec  = _compute_feature_vector(all_norm)
    std_vec  = _standardize(raw_vec)
    x        = std_vec.reshape(1, -1)

    predictions: dict = {}
    confidence:  dict = {}
    for taste in SENSORY_ORDER:
        pred  = float(_models[taste].predict(x)[0])
        pred  = max(0.0, min(100.0, pred))
        rmse  = _rmse.get(taste, 0.0)
        predictions[taste] = round(pred, 1)
        confidence[taste]  = {
            "lower": round(max(0.0, pred - rmse), 1),
            "upper": round(min(100.0, pred + rmse), 1),
        }

    recipe["ingredients"] = all_norm
    return predictions, confidence


# ── Chat system prompt ─────────────────────────────────────────────────────────

_CHAT_SYSTEM = (
    "You are FlavorLab's recipe assistant. You help users explore how ingredient "
    "changes affect the predicted sensory profile (sweet, bitter, salty, umami, sour).\n\n"
    "You can:\n"
    "- Modify ingredient weights/proportions\n"
    "- Add new ingredients\n"
    "- Remove ingredients\n"
    "- Answer questions about the predictions and food science\n\n"
    "CRITICAL: Always respond with ONLY a valid JSON object (no markdown, no code blocks):\n"
    '{"reply":"<natural language>","action":"modify_recipe" or "answer","updated_recipe":null or {...}}\n\n'
    "If modifying (action=modify_recipe): return updated_recipe with the FULL ingredients list. "
    "Keep all existing fields (sensory_scores, source, confidence, evidence, matched_name) unchanged "
    "for existing ingredients. For brand-new ingredients, include only name and weight — the server "
    "will look them up. Weights do NOT need to sum to 1.0; the server normalises automatically.\n\n"
    "If only answering a question (action=answer): set updated_recipe to null.\n\n"
    'updated_recipe format: {"recipe_name":"...","ingredients":[{"name":"...","weight":<float>,...}]}'
)


# ── Chat route ─────────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data       = request.get_json(force=True)
        message    = (data.get("message") or "").strip()
        recipe     = data.get("recipe")        # current enriched recipe (may be None)
        history    = data.get("chat_history") or []
        session_id = data.get("session_id") or str(_uuid.uuid4())

        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Build system prompt with live recipe context
        system = _CHAT_SYSTEM
        if recipe:
            ings  = recipe.get("ingredients", [])
            lines = "\n".join(
                f"  - {i['name']}: {i['weight'] * 100:.1f}%"
                for i in ings
            )
            system += (
                f"\n\nCurrent recipe: {recipe.get('recipe_name', 'Unknown')}\n"
                f"Ingredients:\n{lines}"
            )

        # Build Claude message list from history + new message
        messages = []
        for h in history:
            role    = h.get("role", "user")
            content = h.get("text", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        response = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=messages,
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result         = json.loads(raw)
        reply          = result.get("reply", "I couldn't process that request.")
        action         = result.get("action", "answer")
        updated_recipe = result.get("updated_recipe")

        if action == "modify_recipe" and updated_recipe and updated_recipe.get("ingredients"):
            # Enrich any brand-new ingredients (those without sensory_scores)
            new_ings = [i for i in updated_recipe["ingredients"] if not i.get("sensory_scores")]
            if new_ings:
                enriched     = enrich_ingredients(new_ings)
                enriched_map = {i["name"]: i for i in enriched}
                for ing in updated_recipe["ingredients"]:
                    if not ing.get("sensory_scores") and ing["name"] in enriched_map:
                        ing.update(enriched_map[ing["name"]])

            new_predictions, new_confidence = _run_prediction_from_recipe(updated_recipe)
            return jsonify({
                "reply":       reply,
                "action":      action,
                "recipe":      updated_recipe,
                "predictions": new_predictions,
                "confidence":  new_confidence,
            })

        return jsonify({"reply": reply, "action": "answer"})

    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Could not parse Claude response: {exc}"}), 422
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Session routes ─────────────────────────────────────────────────────────────

@app.route("/session/save", methods=["POST"])
def session_save():
    data       = request.get_json(force=True)
    session_id = data.get("session_id")
    state      = data.get("state", {})
    if session_id:
        _sessions[session_id] = state
    return jsonify({"ok": True})


@app.route("/session/<session_id>", methods=["GET"])
def session_get(session_id):
    state = _sessions.get(session_id)
    if state is None:
        return jsonify({"ok": False}), 404
    return jsonify({"ok": True, "state": state})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
