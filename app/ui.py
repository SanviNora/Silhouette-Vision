"""Silhouette Vision — Streamlit app.

Run: streamlit run app/ui.py

Tabs: search by photo (with garment picker + predicted attributes), search by description,
and a style map of the luxury catalog.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # hosted: no pip install

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from silhouette_vision.config import DATA_REPO, DATA_ROOT, HOSTED, PUBLIC, path
from silhouette_vision.images import load_rgb
from silhouette_vision.search import Filters, SearchEngine

st.set_page_config(page_title="Silhouette Vision", page_icon="👗", layout="wide")

ABOUT = """
Portfolio project, not affiliated with any brand or retailer.
[Code and results](https://github.com/SanviNora/Silhouette-Vision)

**Products (2019–2026)**
- *ZooClaw-Fashion* (2026), SerendipityOne, CC BY-NC 4.0.
- *LookBench* studio gallery (2025), Apache-2.0.
- *Second-Hand Fashion* (2022–24), Nauman et al., RISE, Wargön Innovation and Myrorna,
  CC BY 4.0 (doi:10.5281/zenodo.13788681).
- Footwear: *Amazon Berkeley Objects* (c. 2019–21), Collins et al., CVPR 2022, CC BY 4.0.

**Demand:** *Visuelle 2.0*, Skenderi et al., CVPR Workshops 2022, CC BY-NC-SA 4.0.

**Models:** Marqo-FashionSigLIP, YOLOS-Fashionpedia. Attribute heads trained on Fashion
Product Images (Myntra, MIT).
"""
SOURCE_LABEL = {"zooclaw": "ZooClaw", "lookbench": "LookBench", "secondhand": "Second-hand",
                "abo": "Amazon"}


# "Precise match" loads a second 1.2 GB model: off on hosted/small machines (SILHOUETTE_PRECISE).
PRECISE_ALLOWED = os.environ.get("SILHOUETTE_PRECISE", "0" if HOSTED else "1") == "1"


@st.cache_resource(show_spinner="First start: downloading the demo data (~1 minute)…")
def get_data() -> None:
    """On a fresh host, fetch the public bundle into DATA_ROOT (no-op when the data is there)."""
    if HOSTED or os.environ.get("DATA_REPO"):
        from silhouette_vision.bootstrap import ensure_bundle

        ensure_bundle(DATA_ROOT, DATA_REPO)


@st.cache_resource(show_spinner="Loading model and index…")
def get_engine() -> SearchEngine:
    get_data()
    return SearchEngine("marqo_fashion_siglip", precise=PRECISE_ALLOWED)


@st.cache_resource(show_spinner="Loading garment detector…")
def get_detector():
    from silhouette_vision.detect import GarmentDetector

    return GarmentDetector()


@st.cache_resource(show_spinner=False)
def get_attributes():
    try:
        from silhouette_vision.enrich import AttributePredictor, load_catalog_predictions

        return AttributePredictor(), load_catalog_predictions()
    except FileNotFoundError:
        return None, None


@st.cache_resource(show_spinner=False)
def get_match_confidence():
    try:
        from silhouette_vision.match import MatchConfidence

        return MatchConfidence()
    except FileNotFoundError:
        return None


def show_best_match(engine: SearchEngine, query, filters, precise_query, image=None, query_attrs=None):
    """Highlight the single closest product with a calibrated exact-match likelihood."""
    matcher = get_match_confidence()
    if matcher is None:
        return
    row, top_scores = engine.best_match(query, filters, precise_query)
    prob = matcher.probability(top_scores, "precise" if precise_query is not None else "marqo")
    tier = matcher.tier(prob)
    with st.container(border=True):
        left, right = st.columns([1, 4])
        left.image(str(DATA_ROOT / row.image_path), width="stretch")
        right.markdown(f"#### {tier}")
        right.markdown(f"**{row.brand or ''}** — {row.title}  \n{price_text(row)} · "
                       f"{SOURCE_LABEL[row.source]}{' · Pre-owned' if row.is_preowned else ''}")
        right.metric("Exact-match likelihood", f"{prob:.0%}",
                     help="Calibrated on 2,345 benchmark photos with known products: when this "
                          "says 90%+, the product was the exact one 91% of the time (86% at 80%+). "
                          "Based on how strongly the best result stands out from the next ones, not "
                          "on raw similarity alone.")
        right.caption(f"Visual similarity {row.similarity:.2f} (cosine; 1.00 = identical image)")
        if image is not None:
            show_why(engine, image, query_attrs, row)


@st.cache_resource(show_spinner=False)
def get_recogniser():
    try:
        from silhouette_vision.named_models import ModelRecogniser

        return ModelRecogniser(get_engine().text_vector)
    except FileNotFoundError:
        return None


def show_named_model(engine: SearchEngine, image_vec) -> None:
    """Name an iconic luxury model (e.g. LV Pochette Félicie) even when the catalog lacks it."""
    recogniser = get_recogniser()
    if recogniser is None:
        return
    from urllib.parse import quote_plus

    from silhouette_vision.named_models import catalog_listings

    (model, prob), *_ = recogniser.recognise(image_vec)
    if prob < 0.5:
        return
    with st.container(border=True):
        verdict = "Recognised model" if prob >= 0.8 else "Probably"
        st.markdown(f"#### {verdict}: {model.label}")
        st.metric("Model confidence", f"{prob:.0%}",
                  help="Zero-shot: the photo is compared with the names of 161 iconic luxury models. "
                       "Tested on 2,892 catalog photos: when this says 80%+, the name was right "
                       "96% of the time, and it named a model for 0.3% of other brands' products.")
        rows = catalog_listings(engine.catalog, model)
        web = f"https://www.google.com/search?tbm=shop&q={quote_plus(model.label)}"
        if len(rows) == 0:
            where = ("The public demo searches Myntra, a mass-market catalog that doesn't carry "
                     "this brand." if PUBLIC else "Not stocked in our catalog (a 2019 snapshot).")
            st.markdown(f"{where} [Find it online]({web})")
            return
        best = rows[np.argsort(-(engine.embeddings[rows] @ image_vec))[:4]]
        st.markdown(f"**{len(rows)} listing{'s' if len(rows) > 1 else ''} of this model in our catalog** "
                    f"· [Find it online]({web})")
        cols = st.columns(4)
        for col, (_, row) in zip(cols, engine.catalog.iloc[best].iterrows(), strict=False):
            col.image(str(DATA_ROOT / row.image_path), width="stretch")
            col.caption(f"{row.title}  \n{price_text(row)}{' · Pre-owned' if row.is_preowned else ''}")


@st.cache_data(show_spinner=False)
def get_style_map():
    file = DATA_ROOT / "data/processed/style_clusters.parquet"
    report = path("reports") / "style_clusters.json"
    if not file.exists() or not report.exists():
        return None, None
    return pd.read_parquet(file), json.loads(report.read_text())


def price_text(row) -> str:
    """Only Second-Hand garments carry a price: an estimated resale band in Swedish kronor."""
    band = getattr(row, "price_band", None)
    return f"Resale est. {band} SEK" if isinstance(band, str) and band else ""


def sidebar_filters(engine: SearchEngine) -> tuple[Filters, int, bool]:
    st.sidebar.header("Filters")
    cat = engine.catalog
    sources = st.sidebar.multiselect("Catalog", list(SOURCE_LABEL), format_func=SOURCE_LABEL.get)
    categories = st.sidebar.multiselect("Category", sorted(cat.category.unique()))
    genders = st.sidebar.multiselect("Gender", sorted(cat.gender.dropna().unique()))
    condition = st.sidebar.radio("Condition", ["Any", "New", "Pre-owned"], horizontal=True)
    k = st.sidebar.slider("Results", 4, 48, 12, step=4)
    match_colour = st.sidebar.toggle(
        "Match colour", value=engine.has_colour, disabled=not engine.has_colour,
        help="Re-ranks photo results toward the photo's colours. Improved colour precision "
             "from 42% to 47% in testing, at a ~1-point cost in product-type precision.",
    )
    preowned = {"Any": None, "New": False, "Pre-owned": True}[condition]
    st.sidebar.caption(
        f"{len(cat):,} products, 2019–2026 · ZooClaw-Fashion, LookBench, Second-Hand Fashion, "
        "Amazon Berkeley Objects (shoes). "
        "Model: Marqo-FashionSigLIP."
    )
    with st.sidebar.expander("About & credits"):
        st.markdown(ABOUT)
    return Filters(sources, categories, genders, preowned), k, match_colour


def show_results(results, key_prefix: str, query_attrs=None):
    if results.empty:
        st.info("No products match these filters.")
        return
    from silhouette_vision.enrich import item_tags

    _, preds = get_attributes()
    cols = st.columns(4)
    for i, row in results.iterrows():
        with cols[i % 4]:
            st.image(str(DATA_ROOT / row.image_path), width="stretch")
            badges = " · ".join(
                b for b in [SOURCE_LABEL[row.source], "Pre-owned" if row.is_preowned else None] if b
            )
            st.markdown(f"**{row.brand or ''}**  \n{row.title}")
            tags = item_tags(preds.loc[row.item_id], row.source) if preds is not None else []
            shared = shared_text(query_attrs, row)
            st.caption(f"{price_text(row)}  \n{badges} · {row.category} · similarity {row.similarity:.2f}"
                       + (f"  \n{shared}" if shared else (f"  \n{' · '.join(tags)}" if tags else "")))
            if st.button("More like this", key=f"{key_prefix}-{row.item_id}"):
                st.session_state.similar_to = row.item_id
                st.rerun()


def show_attributes(engine: SearchEngine, image) -> list[dict]:
    predictor, _ = get_attributes()
    if predictor is None:
        return []
    attrs = predictor.predict(engine.image_vector(image))
    if attrs:
        st.markdown("**What we see**")
        st.caption("  ·  ".join(f"{a['label']}: **{a['value']}** ({a['confidence']:.0%})" for a in attrs))
    return attrs


RELATION_MARK = {"same": "✓", "similar": "≈", "different": "✗"}


def shared_text(query_attrs, row) -> str:
    """'Shares: Handbags · Brown' / 'Differs: Printed vs Solid' for a result card."""
    _, preds = get_attributes()
    if not query_attrs or preds is None:
        return ""
    from silhouette_vision.explain import compare_attributes

    comp = compare_attributes(query_attrs, preds.loc[row.item_id], row.source)
    shares = [c["item"] for c in comp if c["relation"] != "different"]
    differs = [f"{c['item']} (yours: {c['photo']})" for c in comp if c["relation"] == "different"]
    return "  \n".join(t for t in [("Shares: " + " · ".join(shares)) if shares else "",
                                    ("Differs: " + " · ".join(differs)) if differs else ""] if t)


def show_why(engine: SearchEngine, image, query_attrs, row) -> None:
    """Attribute comparison and where the match comes from, in both photos."""
    from silhouette_vision.explain import compare_attributes, occlusion_map, overlay

    with st.expander("Why this matches"):
        _, preds = get_attributes()
        if query_attrs and preds is not None:
            comp = compare_attributes(query_attrs, preds.loc[row.item_id], row.source)
            if comp:
                st.markdown("  \n".join(
                    f"{RELATION_MARK[c['relation']]} **{c['label']}**: {c['photo']}"
                    + ("" if c["relation"] == "same" else f" vs {c['item']}") for c in comp))
        if not st.toggle("Show where the match comes from (a few seconds)", key="why_heatmap"):
            return
        item_image = load_rgb(DATA_ROOT / row.image_path)
        with st.spinner("Hiding one region at a time and re-measuring the similarity…"):
            query_vec, item_vec = engine.image_vector(image), engine.item_vector(row.item_id)
            left, right = st.columns(2)
            left.image(overlay(image, occlusion_map(engine.encoder, image, item_vec, grid=8)),
                       caption="Your photo: regions the match depends on", width="stretch")
            right.image(overlay(item_image, occlusion_map(engine.encoder, item_image, query_vec, grid=8)),
                        caption="The match: regions that resemble your photo", width="stretch")
        st.caption("Brighter = hiding this region lowers the visual similarity most (Marqo model). "
                   "Attributes are predicted for both images; ≈ means a neighbouring shade.")


def pick_garment(image, key: str = "search"):
    """Let the user choose the whole photo or one detected item; returns the image to search with."""
    detections = get_detector().items(image)
    if not detections:
        st.image(image, width=180, caption="Your photo (no single garment detected)")
        return image
    crops = [d.crop(image) for d in detections[:5]]
    # Items are shown as thumbnails without a type label: the detector (trained on photos of
    # people) localises well but mislabels product shots (a denim tote -> "skirt" at 96%), and
    # our type classifier mislabels small crops (a chain bag held in hand -> "Bangle"). Once an
    # item is picked, "What we see" describes it with confidences.
    options = ["Whole photo"] + [f"Item {i + 1}" for i in range(len(crops))]
    cols = st.columns(len(options))
    cols[0].image(image, caption="Whole photo", width="stretch")
    for col, crop, label in zip(cols[1:], crops, options[1:], strict=True):
        col.image(crop, caption=label, width="stretch")
    # One item found -> search it; several -> start from the whole photo and let the user choose
    # (on LookBench street photos, auto-picking the top item was worse than no crop at all).
    default = 1 if len(detections) == 1 else 0
    choice = st.radio("Use", options, index=default, horizontal=True, key=f"garment-{key}")
    return image if choice == "Whole photo" else crops[options.index(choice) - 1]


@st.cache_resource(show_spinner="Loading demand model…")
def get_forecaster():
    try:
        from silhouette_vision.demand import DemandForecaster

        return DemandForecaster()
    except FileNotFoundError:
        return None


CLOTHING = {"top", "bottoms", "dress", "outerwear"}
def visuelle_image(image_path: str) -> str:
    """Deploy-bundle thumbnail if present, else the original Visuelle PNG."""
    from pathlib import Path

    thumb = DATA_ROOT / "data/processed/thumbs/visuelle" / Path(image_path).with_suffix(".jpg")
    return str(thumb if thumb.exists() else DATA_ROOT / "data/raw/visuelle2/visuelle2/images" / image_path)


def forecast_tab():
    import datetime

    forecaster = get_forecaster()
    if forecaster is None:
        st.info("Run scripts/08_visuelle_prepare.py and scripts/09_demand.py to build the demand model.")
        return
    st.caption("How many units would a new product sell in its first 12 weeks? Learned from 5,355 "
               "launches of Nuna Lie, an Italian fast-fashion womenswear brand (110 stores, 2017–2019; "
               "Visuelle 2.0, CC BY-NC-SA). A new product has no sales history, so the model borrows "
               "from past products that look like it.")
    upload = st.file_uploader("Upload a photo of the new product", type=["jpg", "jpeg", "png", "webp"],
                              key="forecast-upload")
    if not upload:
        return
    engine = get_engine()
    image = pick_garment(load_rgb(upload), key="forecast")
    vec = engine.image_vector(image)

    predictor, _ = get_attributes()
    if predictor is not None:
        attrs = predictor.predict(vec)
        kind = next((a["value"] for a in attrs if a["attribute"] == "article_type"), None)
        category = predictor.type_to_category.get(kind)
        if category is not None and category not in CLOTHING:
            st.warning(f"This looks like **{kind}**. The brand in this dataset sells women's clothing "
                       "only, so there is no sales history to forecast from.")
            return

    b = forecaster.b
    tags = forecaster.suggest_tags(vec)
    left, mid, right = st.columns(3)
    category = left.selectbox("Category", b["categories"]["category"],
                              index=b["categories"]["category"].index(tags["category"]))
    colour = mid.selectbox("Colour", b["categories"]["color"],
                           index=b["categories"]["color"].index(tags["color"]))
    fabric = right.selectbox("Fabric", b["categories"]["fabric"],
                             index=b["categories"]["fabric"].index(tags["fabric"]))
    st.caption("Suggested from the most similar past products (right about 2 times in 3 for colour "
               "and fabric, 3 in 4 for category); change them if they're wrong.")
    left, mid, right = st.columns(3)
    n_stores = left.slider("Number of stores", 1, len(b["store_order"]),
                           int(np.median(b["latest_n_stores"])),
                           help="Stores are the brand's most-used launch stores. Distribution breadth "
                                "is the strongest single predictor: it reflects the planners' own "
                                "expectations.")
    price_pct = mid.slider("Price level (vs the brand's range)", 0, 100, 50, 5,
                           help="Percentile of the brand's latest-season prices (real prices are "
                                "anonymised in the dataset).") / 100
    launch = right.date_input("Launch date", datetime.date(2019, 9, 2))

    r = forecaster.forecast(vec, category, colour, fabric, price_pct, n_stores, launch)
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected sales, 12 weeks", f"{r['total']:,.0f} units")
    c2.metric("Likely range (80%)", f"{r['low']:,.0f} – {r['high']:,.0f}",
              help="On 1,900 unseen products from the 2019 seasons, actual 12-week sales were "
                   "within this range of the prediction for 80% of products.")
    c3.metric("Per store", f"{r['per_store']:.1f} units")
    weeks = pd.DataFrame({"Week": np.arange(1, 13), "Units": r["weekly"]})
    st.plotly_chart(px.bar(weeks, x="Week", y="Units", height=260), width="stretch")
    similarity = forecaster.nearest_similarity(vec)
    if similarity < b["min_similarity"]:
        st.caption(f"Closest past product similarity {similarity:.2f}; the brand's own new products "
                   f"score ≥ {b['min_similarity']:.2f} (flat product shots). A photo of a person or a "
                   "busy background lowers this, so treat the forecast as rough.")

    st.markdown("**Borrowed from: the most similar past launches**")
    cols = st.columns(8)
    for col, (_, row) in zip(cols, r["lookalikes"].iterrows(), strict=False):
        col.image(visuelle_image(row.image_path), width="stretch")
        col.caption(f"{row.category} · {row.color}  \n{row.units_per_store:.1f} units/store · "
                    f"{row.n_stores} stores · {row.season}")
    st.caption("Model: gradient boosting on store, launch timing, tags, price, distribution breadth "
               "and look-alike sales. Tested on the 2019 seasons (never seen in training): weekly "
               "product sales error 34.6% (WAPE) vs 43.7% for a seasonal average.")


def style_map_tab():
    clusters, report = get_style_map()
    if clusters is None:
        st.info("Run scripts/05_style_clusters.py to build the style map.")
        return
    engine = get_engine()
    st.caption("Visual style clusters in the Farfetch luxury catalog, found from images alone "
               "(k-means on UMAP of Marqo embeddings) and named by the model's most distinctive "
               "style words. Prices in SGD.")
    category = st.selectbox("Category", list(report), index=list(report).index("bag"))
    data = clusters[clusters.category == category]
    info = report[category]
    summary = pd.DataFrame([{"Style": v["name"], "Items": v["size"],
                             "Median price": v["median_price_sgd"],
                             "On sale": v["on_sale_share"], "Pre-owned": v["preowned_share"],
                             "Signature brands": ", ".join(v["signature_brands"])}
                            for v in info["detail"].values()]).sort_values("Median price", ascending=False)
    sample = data.sample(min(6000, len(data)), random_state=0)
    if PUBLIC:  # the Farfetch listings may not be redistributed: map positions and styles only
        hover = {"map_x": False, "map_y": False}
    else:
        sample = sample.merge(engine.catalog[["item_id", "brand", "title", "price"]], on="item_id")
        hover = {"brand": True, "title": True, "price": ":.0f", "map_x": False, "map_y": False}
    fig = px.scatter(sample, x="map_x", y="map_y", color="cluster_name", opacity=0.6,
                     hover_data=hover, labels={"cluster_name": "Style"}, height=560)
    fig.update_traces(marker={"size": 4})
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, legend_title_text="Style")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "Median price": st.column_config.NumberColumn(format="S$%.0f"),
        "On sale": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Pre-owned": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    })
    if PUBLIC:
        st.caption("Aggregates over a 2019 Farfetch listings snapshot, used for analysis only; "
                   "individual products are not shown in the public demo.")
        return
    style = st.selectbox("Show products from a style", summary.Style.tolist())
    members = data[data.cluster_name == style].sample(
        min(8, (data.cluster_name == style).sum()), random_state=1)
    cols = st.columns(8)
    for col, item_id in zip(cols, members.item_id, strict=False):
        row = engine.catalog.set_index("item_id").loc[item_id]
        col.image(str(DATA_ROOT / row.image_path), caption=f"{row.brand}", width="stretch")


def main():
    engine = get_engine()
    filters, k, match_colour = sidebar_filters(engine)

    st.title("Silhouette Vision")
    st.caption("Search fashion products by photo, by description, or both.")

    if item_id := st.session_state.get("similar_to"):
        row = engine.catalog.set_index("item_id").loc[item_id]
        left, right = st.columns([1, 4])
        left.image(str(DATA_ROOT / row.image_path), width="stretch")
        right.subheader("More like this")
        right.write(f"**{row.brand or ''}** — {row.title}")
        if right.button("Clear"):
            del st.session_state["similar_to"]
            st.rerun()
        colour = engine.item_colour(item_id) if match_colour else None
        show_results(engine.search(engine.item_vector(item_id), k, filters, exclude=[item_id],
                                   query_colour=colour), "sim")
        return

    photo_tab, text_tab, map_tab, forecast = st.tabs(
        ["Search by photo", "Search by description", "Style map", "New product forecast"])

    with photo_tab:
        upload = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png", "webp"])
        refine = st.text_input("Optional: refine with text", placeholder="e.g. in red, leather, cropped")
        weight = st.slider("Text influence", 0.0, 0.8, 0.3, 0.05, disabled=not refine)
        precise = engine.has_precise and PRECISE_ALLOWED and st.toggle(
            "Precise match (Myntra)",
            help="Blends Marqo with GR-Lite, a fashion-retrieval model. In testing: +4.7 points "
                 "exact-match recall overall and +7.6 on street photos. Searches the Myntra "
                 "catalog only, and is slower (loads a second model on first use).")
        if upload:
            image = pick_garment(load_rgb(upload), key="search")
            query_attrs = show_attributes(engine, image)
            query = engine.image_vector(image)
            show_named_model(engine, query)
            if refine:
                query = engine.combine(query, engine.text_vector(refine), weight)
            colour = engine.image_colour(image) if match_colour else None
            precise_query = None
            if precise:
                with st.spinner("Precise match…"):
                    precise_query = engine.precise_vector(image)
            show_best_match(engine, query, filters, precise_query, image, query_attrs)
            st.markdown("**Similar products**")
            show_results(engine.search(query, k, filters, query_colour=colour,
                                       precise_query=precise_query), "img", query_attrs)

    with text_tab:
        text = st.text_input("Describe what you're looking for",
                             placeholder="e.g. black quilted leather shoulder bag with gold chain")
        if text:
            show_results(engine.search(engine.text_vector(text), k, filters), "txt")

    with map_tab:
        style_map_tab()

    with forecast:
        forecast_tab()


main()
