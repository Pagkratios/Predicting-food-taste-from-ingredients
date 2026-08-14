#!/usr/bin/env python3
"""
Explicit, name-exact ingredient class definitions for the umami-synergy test.

Classes are defined by EXACT ingredient-name match against the 115 unique
ingredient names present in data/raw_recipes.py (a verbatim copy of the
repo dataset).  Exact matching (not regex/keyword matching) is used
deliberately so that every assignment is auditable one-by-one; the repo's
own chemistry features use regex keyword matching, which is retained
unchanged for the 8 baseline features.

Rationale for each judgement call is in AMBIGUOUS_NOTES and is reproduced
verbatim in REPORT.md.
"""

# ─────────────────────────────────────────────────────────────────────
# NUCLEOTIDE sources (IMP / GMP)
# Spec: beef, pork, chicken, other meat, poultry, fish, shellfish,
#       anchovy, mushroom, yeast, yeast extract, bouillon/stock cubes.
# Dataset contains NO mushroom, shellfish, anchovy (as such), fish sauce,
# yeast extract, or bouillon/stock cube entries.
# ─────────────────────────────────────────────────────────────────────
NUCLEOTIDE = [
    "Chicken fillet raw",
    "Cod boiled",
    "Ham lean boiled",
    "Minced beef raw",
    "Minced beef/pork raw",
    "Pork fillet raw",
    "Pork fillet raw (69%) + Beef (31%)",
    "Yeast",
]

# ─────────────────────────────────────────────────────────────────────
# GLUTAMATE sources (free Glu) - PRIMARY (strict) list
# Spec: tomato and tomato products, aged/hard cheese, soy sauce, fish
#       sauce, vinegar, mustard, other fermented items, onion, garlic,
#       leek, seaweed, peas, corn.
# Dataset contains NO fish sauce, seaweed, or corn entries.
# ─────────────────────────────────────────────────────────────────────
GLUTAMATE = [
    # tomato + tomato products
    "Tomato average raw",
    "Tomato puree/concentrate",
    # aged / hard cheese
    "Cheese 20+",
    "Cheese Gouda 48+ average",
    # soy sauce
    "Soy sauce Kikkoman",
    # vinegar, mustard
    "Vinegar",
    "Mustard",
    # other fermented items
    "Sauerkraut (zuurkool)",
    "Spices/Worcestershire",
    # alliums
    "Onions raw",
    "Garlic",
    "Leek boiled",
    "Spring onion",
    "Spices/onion",
    "Herbs (parsley/garlic)",
    # peas
    "Peas frozen boiled",
    "Peas garden super fine tinned",
]

# Sensitivity ("broad") glutamate list: PRIMARY plus every item that was a
# genuine coin-flip.  Used only for the Step-5 sensitivity variant so the
# reader can see whether the verdict depends on these calls.
GLUTAMATE_BROAD = GLUTAMATE + [
    "Cheese Mozzarella",
    "Cheese cream soft Mon Chou",
    "Yoghurt full fat",
    "Capers",
    "Olives tinned/glass",
    "Spices/sambal",
    "Peanut sauce jar prepared",
]

AMBIGUOUS_NOTES = [
    ("Beef gelatin", "NUCLEOTIDE",
     "EXCLUDED. Gelatin is hydrolysed collagen; the manufacturing process "
     "removes essentially all nucleotides. It is a meat-derived item but not "
     "an IMP source."),
    ("Eggs chicken boiled average", "NUCLEOTIDE",
     "EXCLUDED. Egg is animal protein but not a recognised IMP/GMP source; "
     "the spec lists meat/poultry flesh, fish, mushroom, yeast."),
    ("Leavening/sugar", "NUCLEOTIDE",
     "EXCLUDED. Ambiguous label. Read as chemical leavening + sugar rather "
     "than baker's yeast. Yeast in baked goods is also spent/inactivated."),
    ("Toast / Baguette white / Flour*", "NUCLEOTIDE",
     "EXCLUDED. Bread is yeast-leavened but the yeast is spent and present "
     "at trace mass; counting all bread as a nucleotide source would put "
     "f_nucleotide > 0 in nearly every recipe and destroy the contrast."),
    ("Whey", "NUCLEOTIDE",
     "EXCLUDED. Dairy protein fraction, not a nucleotide source."),
    ("Cheese 20+", "GLUTAMATE",
     "INCLUDED. Dutch '20+' denotes fat-in-dry-matter of a ripened "
     "Gouda-type cheese, i.e. aged/hard cheese."),
    ("Cheese Mozzarella", "GLUTAMATE",
     "EXCLUDED from primary (in broad). Fresh pasta-filata cheese; free "
     "glutamate is roughly an order of magnitude below aged Gouda. The spec "
     "says 'aged/hard cheese'."),
    ("Cheese cream soft Mon Chou", "GLUTAMATE",
     "EXCLUDED from primary (in broad). Fresh unripened cream cheese."),
    ("Yoghurt full fat", "GLUTAMATE",
     "EXCLUDED from primary (in broad). Fermented, but lactic fermentation "
     "of milk yields little free glutamate compared with proteolytic "
     "ripening."),
    ("Spices/Worcestershire", "GLUTAMATE",
     "INCLUDED as GLUTAMATE. Genuinely dual: Worcestershire sauce is a "
     "fermented condiment (tamarind/vinegar/molasses) that also contains "
     "anchovy, so it arguably belongs to BOTH classes. Assigned to "
     "GLUTAMATE only, matching the repo's own FERMENT_PAT which already "
     "lists 'worcestershire'. Assigning it to both classes would make a "
     "single ingredient generate synergy on its own."),
    ("Spices/onion, Herbs (parsley/garlic)", "GLUTAMATE",
     "INCLUDED. Composite spice/herb labels whose stated composition "
     "contains an allium. Matches the repo's own ALLIUM_PAT behaviour."),
    ("Capers, Olives tinned/glass", "GLUTAMATE",
     "EXCLUDED from primary (in broad). Brine-cured; some lactic "
     "fermentation occurs but they are not conventionally cited as free-Glu "
     "sources and appear at very small mass fractions."),
    ("Spices/sambal", "GLUTAMATE",
     "EXCLUDED from primary (in broad). Some sambals are fermented, most "
     "commercial ones are not; composition unknown from the label."),
    ("Peanut sauce jar prepared", "GLUTAMATE",
     "EXCLUDED from primary (in broad). Commercial satay sauce commonly "
     "contains soy sauce and onion (and often added MSG), but the label "
     "gives no composition, so this is a guess either way."),
    ("Beans brown tinned, Broccoli boiled, Kale curly boiled, "
     "White cabbage (raw), Water chestnut, Carrots*, Celery",
     "GLUTAMATE",
     "EXCLUDED. Vegetables not named in the spec's glutamate list. Some "
     "(e.g. Chinese cabbage) do carry free Glu, but including arbitrary "
     "vegetables would make the class unfalsifiable."),
]


def validate(all_names):
    """Fail loudly if a class member is not an actual dataset ingredient."""
    known = set(all_names)
    missing = [n for n in NUCLEOTIDE + GLUTAMATE_BROAD if n not in known]
    if missing:
        raise ValueError(f"Class members not present in dataset: {missing}")
    overlap = set(NUCLEOTIDE) & set(GLUTAMATE_BROAD)
    if overlap:
        raise ValueError(f"Ingredient assigned to both classes: {overlap}")
    return True
