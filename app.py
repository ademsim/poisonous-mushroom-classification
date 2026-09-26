from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Mushroom Classifier")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "mushroom_model.pkl", "rb"))


model = load_model()
cat_cols = list(model.named_steps["prep"].transformers_[0][2])
categories = model.named_steps["prep"].named_transformers_["cat"].categories_
known = {c: [v for v in cats if isinstance(v, str)] for c, cats in zip(cat_cols, categories)}

COLOR = {"n": "brown", "b": "buff", "g": "gray", "r": "green", "p": "pink", "u": "purple", "e": "red",
         "w": "white", "y": "yellow", "l": "blue", "o": "orange", "k": "black", "f": "none"}
SURFACE = {"i": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth", "h": "shiny", "l": "leathery",
           "k": "silky", "t": "sticky", "w": "wrinkled", "e": "fleshy", "f": "none"}
LEGEND = {
    "cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "s": "sunken", "p": "spherical", "o": "other"},
    "cap-surface": SURFACE,
    "cap-color": COLOR,
    "does-bruise-or-bleed": {"t": "bruises or bleeds", "f": "no"},
    "gill-attachment": {"a": "adnate", "x": "adnexed", "d": "decurrent", "e": "free", "s": "sinuate", "p": "pores", "f": "none"},
    "gill-spacing": {"c": "close", "d": "distant", "f": "none"},
    "gill-color": COLOR,
    "stem-root": {"b": "bulbous", "s": "swollen", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs", "r": "rooted"},
    "stem-surface": SURFACE,
    "stem-color": COLOR,
    "veil-type": {"p": "partial", "u": "universal"},
    "veil-color": COLOR,
    "has-ring": {"t": "has a ring", "f": "no ring"},
    "ring-type": {"c": "cobwebby", "e": "evanescent", "r": "flaring", "g": "grooved", "l": "large", "p": "pendant",
                  "s": "sheathing", "z": "zone", "y": "scaly", "m": "movable", "f": "none"},
    "spore-print-color": COLOR,
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "h": "heaths", "u": "urban", "w": "waste", "d": "woods"},
    "season": {"s": "spring", "u": "summer", "a": "autumn", "w": "winter"},
}

st.title("Poisonous Mushroom Classifier")
st.write(
    "A gradient boosting model (trained on the Kaggle Poisonous Mushrooms dataset, Playground Series S4E8) predicts "
    "whether a mushroom is edible or poisonous from its physical characteristics. Leave a feature as "
    "'unknown' if you cannot tell."
)
st.warning("Educational demo only. Never decide whether a wild mushroom is safe to eat based on a model.")

c1, c2, c3 = st.columns(3)
diameter = c1.slider("Cap diameter (cm)", 0.5, 30.0, 6.0, 0.1)
height = c2.slider("Stem height (cm)", 0.0, 30.0, 6.0, 0.1)
width = c3.slider("Stem width (mm)", 0.0, 60.0, 10.0, 0.5)

values = {}
columns = st.columns(3)
for i, col in enumerate(cat_cols):
    codes = known[col]
    label = col.replace("-", " ").capitalize()
    choice = columns[i % 3].selectbox(
        label,
        [None] + codes,
        format_func=lambda c, col=col: "unknown" if c is None else f"{LEGEND[col].get(c, '?')} ({c})",
        key=col,
    )
    values[col] = np.nan if choice is None else choice

if st.button("Classify"):
    row = {**values, "cap-diameter": diameter, "stem-height": height, "stem-width": width}
    X = pd.DataFrame([row])[list(model.feature_names_in_)]
    X[cat_cols] = X[cat_cols].astype(object)
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    p_poison = float(proba["p"])
    if p_poison >= 0.5:
        st.error(f"☠️ Predicted: **poisonous** ({p_poison:.1%} probability)")
    else:
        st.success(f"✅ Predicted: **edible** ({1 - p_poison:.1%} probability)")
    st.progress(p_poison)
    st.caption("The bar shows the probability of being poisonous.")

st.caption(
    "Model: HistGradientBoosting pipeline (validation MCC ≈ 0.98 on the Kaggle data). The Kaggle data is "
    "synthetic and derived from the UCI Secondary Mushroom dataset, so this is a modeling exercise, not field advice."
)
