"""Silhouette Vision — Streamlit app.

Run: streamlit run app/ui.py

Tabs: search by photo (with garment picker + predicted attributes), search by description,
and a style map of the luxury catalog.
"""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from silhouette_vision.config import ROOT, path
from silhouette_vision.images import load_rgb
from silhouette_vision.search import Filters, SearchEngine

st.set_page_config(page_title="Silhouette Vision", page_icon="👗", layout="wide")

CURRENCY = {"SGD": "S$", "INR": "₹"}
SOURCE_LABEL = {"myntra": "Myntra", "farfetch": "Farfetch"}


@st.cache_resource(show_spinner="Loading model and index…")
def get_engine() -> SearchEngine:
    return SearchEngine("marqo_fashion_siglip")


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
        left.image(str(ROOT / row.image_path), width="stretch")
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
            st.markdown(f"Not stocked in our catalog (a 2019 snapshot). [Find it online]({web})")
            return
        best = rows[np.argsort(-(engine.embeddings[rows] @ image_vec))[:4]]
        st.markdown(f"**{len(rows)} listing{'s' if len(rows) > 1 else ''} of this model in our catalog** "
                    f"· [Find it online]({web})")
        cols = st.columns(4)
        for col, (_, row) in zip(cols, engine.catalog.iloc[best].iterrows(), strict=False):
            col.image(str(ROOT / row.image_path), width="stretch")
            col.caption(f"{row.title}  \n{price_text(row)}{' · Pre-owned' if row.is_preowned else ''}")


@st.cache_data(show_spinner=False)
def get_style_map():
    file = ROOT / "data/processed/style_clusters.parquet"
    report = path("reports") / "style_clusters.json"
    if not file.exists() or not report.exists():
        return None, None
    return pd.read_parquet(file), json.loads(report.read_text())


def price_text(row) -> str:
    if row.price != row.price:  # NaN
        return ""
    text = f"{CURRENCY.get(row.currency, '')}{row.price:,.0f}"
    if row.on_sale and row.discount_pct and row.discount_pct > 0:
        text += f" · −{row.discount_pct:.0%}"
    return text


def sidebar_filters(engine: SearchEngine) -> tuple[Filters, int, bool]:
    st.sidebar.header("Filters")
    cat = engine.catalog
    sources = st.sidebar.multiselect("Catalog", ["myntra", "farfetch"], format_func=SOURCE_LABEL.get)
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
        f"{len(cat):,} products · Myntra (MIT) + Farfetch (2019 scrape, prices in SGD). "
        "Model: Marqo-FashionSigLIP."
    )
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
            st.image(str(ROOT / row.image_path), width="stretch")
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
        item_image = load_rgb(ROOT / row.image_path)
        with st.spinner("Hiding one region at a time and re-measuring the similarity…"):
            query_vec, item_vec = engine.image_vector(image), engine.item_vector(row.item_id)
            left, right = st.columns(2)
            left.image(overlay(image, occlusion_map(engine.encoder, image, item_vec, grid=8)),
                       caption="Your photo: regions the match depends on", width="stretch")
            right.image(overlay(item_image, occlusion_map(engine.encoder, item_image, query_vec, grid=8)),
                        caption="The match: regions that resemble your photo", width="stretch")
        st.caption("Brighter = hiding this region lowers the visual similarity most (Marqo model). "
                   "Attributes are predicted for both images; ≈ means a neighbouring shade.")


def pick_garment(image):
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
    choice = st.radio("Search for", options, index=default, horizontal=True)
    return image if choice == "Whole photo" else crops[options.index(choice) - 1]


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
    sample = data.sample(min(6000, len(data)), random_state=0).merge(
        engine.catalog[["item_id", "brand", "title", "price"]], on="item_id")
    fig = px.scatter(sample, x="map_x", y="map_y", color="cluster_name", opacity=0.6,
                     hover_data={"brand": True, "title": True, "price": ":.0f",
                                 "map_x": False, "map_y": False},
                     labels={"cluster_name": "Style"}, height=560)
    fig.update_traces(marker={"size": 4})
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, legend_title_text="Style")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "Median price": st.column_config.NumberColumn(format="S$%.0f"),
        "On sale": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Pre-owned": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    })
    style = st.selectbox("Show products from a style", summary.Style.tolist())
    members = data[data.cluster_name == style].sample(
        min(8, (data.cluster_name == style).sum()), random_state=1)
    cols = st.columns(8)
    for col, item_id in zip(cols, members.item_id, strict=False):
        row = engine.catalog.set_index("item_id").loc[item_id]
        col.image(str(ROOT / row.image_path), caption=f"{row.brand}", width="stretch")


def main():
    engine = get_engine()
    filters, k, match_colour = sidebar_filters(engine)

    st.title("Silhouette Vision")
    st.caption("Search fashion products by photo, by description, or both.")

    if item_id := st.session_state.get("similar_to"):
        row = engine.catalog.set_index("item_id").loc[item_id]
        left, right = st.columns([1, 4])
        left.image(str(ROOT / row.image_path), width="stretch")
        right.subheader("More like this")
        right.write(f"**{row.brand or ''}** — {row.title}")
        if right.button("Clear"):
            del st.session_state["similar_to"]
            st.rerun()
        colour = engine.item_colour(item_id) if match_colour else None
        show_results(engine.search(engine.item_vector(item_id), k, filters, exclude=[item_id],
                                   query_colour=colour), "sim")
        return

    photo_tab, text_tab, map_tab = st.tabs(["Search by photo", "Search by description", "Style map"])

    with photo_tab:
        upload = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png", "webp"])
        refine = st.text_input("Optional: refine with text", placeholder="e.g. in red, leather, cropped")
        weight = st.slider("Text influence", 0.0, 0.8, 0.3, 0.05, disabled=not refine)
        precise = engine.has_precise and st.toggle(
            "Precise match (Myntra)",
            help="Blends Marqo with GR-Lite, a fashion-retrieval model. In testing: +4.7 points "
                 "exact-match recall overall and +7.6 on street photos. Searches the Myntra "
                 "catalog only, and is slower (loads a second model on first use).")
        if upload:
            image = pick_garment(load_rgb(upload))
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


main()
