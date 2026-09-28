"""Silhouette Vision — Streamlit app.

Run: streamlit run app/ui.py

Tabs: search by photo (with garment picker + predicted attributes), search by description,
and a style map of the luxury catalog.
"""

import json

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


def show_results(results, key_prefix: str):
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
            st.caption(f"{price_text(row)}  \n{badges} · {row.category} · similarity {row.similarity:.2f}"
                       + (f"  \n{' · '.join(tags)}" if tags else ""))
            if st.button("More like this", key=f"{key_prefix}-{row.item_id}"):
                st.session_state.similar_to = row.item_id
                st.rerun()


def show_attributes(engine: SearchEngine, image):
    predictor, _ = get_attributes()
    if predictor is None:
        return
    attrs = predictor.predict(engine.image_vector(image))
    if attrs:
        st.markdown("**What we see**")
        st.caption("  ·  ".join(f"{a['label']}: **{a['value']}** ({a['confidence']:.0%})" for a in attrs))


def pick_garment(image):
    """Let the user choose the whole photo or one detected item; returns the image to search with."""
    detections = get_detector().detect(image)
    if not detections:
        st.image(image, width=180, caption="Your photo (no single garment detected)")
        return image
    crops = [d.crop(image) for d in detections[:5]]
    options = ["Whole photo"] + [f"{d.name} ({d.score:.0%})" for d in detections[:5]]
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
        if upload:
            image = pick_garment(load_rgb(upload))
            show_attributes(engine, image)
            query = engine.image_vector(image)
            if refine:
                query = engine.combine(query, engine.text_vector(refine), weight)
            colour = engine.image_colour(image) if match_colour else None
            show_results(engine.search(query, k, filters, query_colour=colour), "img")

    with text_tab:
        text = st.text_input("Describe what you're looking for",
                             placeholder="e.g. black quilted leather shoulder bag with gold chain")
        if text:
            show_results(engine.search(engine.text_vector(text), k, filters), "txt")

    with map_tab:
        style_map_tab()


main()
