
# =========================================================
# Supply Chain Intelligence Dashboard
# =========================================================

import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Supply Chain Intelligence Dashboard",
    page_icon="📦",
    layout="wide",
)

# =========================================================
# PROJECT CONFIGURATION
# =========================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

os.chdir(PROJECT_ROOT)

DATA_DIR   = PROJECT_ROOT / "data"  / "clean"
MODEL_DIR  = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

PREDICTIONS_FILE       = DATA_DIR   / "ml_predictions.csv"
BASELINE_METRICS_FILE  = REPORT_DIR / "baseline_metrics.json"
ENRICHED_METRICS_FILE  = REPORT_DIR / "enriched_metrics.json"
BASELINE_PIPELINE_FILE = MODEL_DIR  / "baseline_pipeline.joblib"
ENRICHED_PIPELINE_FILE = MODEL_DIR  / "enriched_pipeline.joblib"
GRAPH_FEATURES_FILE    = DATA_DIR   / "graph_features.csv"

QDRANT_COLLECTION    = "supply_chain_products"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# =========================================================
# ENV / CREDENTIALS
# =========================================================

from dotenv import load_dotenv
load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", 6333))

# =========================================================
# PAGE HEADER
# =========================================================

st.title("📦 Supply Chain Intelligence Dashboard")
st.caption(
    "DuckDB cleans. Neo4j connects. "
    "GDS scores. pandas merges. "
    "scikit-learn measures. Streamlit ships."
)

# =========================================================
# CACHED RESOURCE LOADERS
# =========================================================

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_baseline_pipeline():
    return joblib.load(BASELINE_PIPELINE_FILE)

@st.cache_resource
def load_enriched_pipeline():
    return joblib.load(ENRICHED_PIPELINE_FILE)

# =========================================================
# NEO4J — plain helper, NOT cached (avoids cache conflict)
# =========================================================

def run_neo4j_query(query, params=None):
    """
    Create a fresh driver, run the query, close the driver.
    Never cached — the DataFrame results are cached instead.
    """
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    try:
        with driver.session(database="neo4j") as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    finally:
        driver.close()

# =========================================================
# DATA LOADERS — cache the DataFrame, not the Neo4j call
# =========================================================

MERGED_DF_FILE = DATA_DIR / "merged_df.csv"

@st.cache_data(ttl=300)
def load_prediction_data():
    return pd.read_csv(PREDICTIONS_FILE)

@st.cache_data(ttl=300)
def load_merged_df():
    """Load full merged dataset for real order lookup by product name."""
    df = pd.read_csv(MERGED_DF_FILE)
    df["Order Id"] = df["Order Id"].astype(str)
    return df

@st.cache_data(ttl=300)
def load_model_metrics():
    with open(BASELINE_METRICS_FILE) as f:
        baseline = json.load(f)
    with open(ENRICHED_METRICS_FILE) as f:
        enriched = json.load(f)
    return baseline, enriched

@st.cache_data(ttl=300)
def load_graph_features_csv():
    """
    PRIMARY SOURCE: load pagerank and community from the
    graph_features.csv saved by 03_graph_analytics.ipynb.
    This is the reliable fallback when Neo4j properties
    are not yet queryable via MATCH.
    """
    df = pd.read_csv(GRAPH_FEATURES_FILE)
    df["order_id"]  = df["order_id"].astype(str)
    df["pagerank"]  = df["pagerank"].astype(float).round(6)
    df["community"] = df["community"].astype(str)
    return df

@st.cache_data(ttl=300)
def load_top_pagerank_orders(limit=10):
    """
    Try Neo4j first. Fall back to graph_features.csv
    if Neo4j returns empty (properties not yet written).
    """
    # Try Neo4j
    try:
        rows = run_neo4j_query("""
            MATCH (o:Order)
            WHERE o.pagerank IS NOT NULL
            RETURN
                o.id       AS order_id,
                o.pagerank AS pagerank,
                o.community AS community
            ORDER BY pagerank DESC
            LIMIT $limit
        """, {"limit": limit})

        if rows:
            df = pd.DataFrame(rows)
            df["pagerank"]  = df["pagerank"].astype(float).round(6)
            df["community"] = df["community"].astype(str)
            return df
    except Exception:
        pass

    # Fallback: read from CSV saved by NB03
    try:
        df = load_graph_features_csv()
        return (
            df[["order_id", "pagerank", "community"]]
            .sort_values("pagerank", ascending=False)
            .head(limit)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame(columns=["order_id", "pagerank", "community"])

@st.cache_data(ttl=300)
def load_community_distribution():
    """
    Try Neo4j first. Fall back to graph_features.csv
    if Neo4j returns empty (properties not yet written).
    """
    # Try Neo4j
    try:
        rows = run_neo4j_query("""
            MATCH (o:Order)
            WHERE o.community IS NOT NULL
            RETURN
                toString(o.community) AS community,
                COUNT(o)              AS order_count
            ORDER BY order_count DESC
            LIMIT 10
        """)

        if rows:
            df = pd.DataFrame(rows)
            df["community"]   = df["community"].astype(str)
            df["order_count"] = df["order_count"].astype(int)
            return df
    except Exception:
        pass

    # Fallback: aggregate from CSV saved by NB03
    try:
        df = load_graph_features_csv()
        comm_df = (
            df.groupby("community", as_index=False)
              .agg(order_count=("order_id", "count"))
              .sort_values("order_count", ascending=False)
              .head(10)
              .reset_index(drop=True)
        )
        return comm_df
    except Exception:
        return pd.DataFrame(columns=["community", "order_count"])

# =========================================================
# QDRANT SIMILARITY SEARCH
# =========================================================

def perform_similarity_search(query_text, top_k=5, category_filter=None):
    model  = load_embedding_model()
    client = get_qdrant_client()

    query_vector = model.encode(
        query_text, normalize_embeddings=True
    ).tolist()

    qdrant_filter = None
    if category_filter and category_filter.strip():
        qdrant_filter = Filter(
            must=[FieldCondition(
                key="category",
                match=MatchValue(value=category_filter.strip()),
            )]
        )

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=qdrant_filter,
    )

    rows = []
    for hit in results.points:
        p = hit.payload
        rows.append({
            "similarity_score":    round(hit.score, 4),
            "product_name":        p.get("product_name",  "N/A"),
            "category":            p.get("category",      "N/A"),
            "department":          p.get("department",    "N/A"),
            "avg_late_risk":       p.get("avg_late_risk", "N/A"),
            "avg_risk_probability":p.get("avg_risk_proba","N/A"),
            "avg_delay_days":      p.get("avg_delay",     "N/A"),
            "order_count":         p.get("order_count",   "N/A"),
        })
    return pd.DataFrame(rows)

# =========================================================
# ML PREDICTION
# =========================================================

def predict_order_risk(order_features):
    pipeline    = load_enriched_pipeline()
    input_df    = pd.DataFrame([order_features])
    prediction  = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    return int(prediction), round(float(probability), 4)

# =========================================================
# LOAD MAIN DATA
# =========================================================

prediction_df = load_prediction_data()

# =========================================================
# TABS
# =========================================================

(
    tab_kpi,
    tab_model,
    tab_graph,
    tab_similarity,
    tab_prediction,
) = st.tabs([
    "📈 Business KPIs",
    "📊 Model Performance",
    "🔗 Graph Analytics",
    "🔍 Similarity Search",
    "⚠️ Order Risk Predictor",
])

# =========================================================
# TAB 1 — BUSINESS KPI OVERVIEW
# =========================================================

with tab_kpi:

    st.header("Business KPI Overview")

    total_orders             = len(prediction_df)
    average_risk_probability = prediction_df["predicted_proba"].mean()
    high_risk_orders         = len(prediction_df[prediction_df["predicted_proba"] > 0.7])
    predicted_late_orders    = int(prediction_df["predicted_risk"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders",              f"{total_orders:,}")
    col2.metric("Avg Risk Probability",      f"{average_risk_probability:.2%}")
    col3.metric("High Risk Orders (>70%)",   f"{high_risk_orders:,}")
    col4.metric("Predicted Late Deliveries", f"{predicted_late_orders:,}")

    st.markdown("---")
    st.subheader("Late Delivery Risk Distribution")

    risk_fig = px.histogram(
        prediction_df,
        x="predicted_proba",
        nbins=40,
        title="Distribution of Predicted Late-Delivery Risk Probabilities",
        labels={"predicted_proba": "Late Delivery Probability"},
        color_discrete_sequence=["#e05c5c"],
    )
    risk_fig.update_layout(bargap=0.05)
    st.plotly_chart(risk_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 20 Highest-Risk Orders")

    high_risk_df = prediction_df.sort_values(
        by="predicted_proba", ascending=False
    ).head(20)
    st.dataframe(high_risk_df, use_container_width=True)

# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================

with tab_model:

    st.header("Baseline vs Graph-Enriched Model")
    st.markdown("""
    Comparing two models trained in Notebook 04:
    - **Baseline (S5):** tabular features + node degree only
    - **Graph-Enriched (S6):** baseline + PageRank + community (from Neo4j GDS)
    """)

    try:
        baseline_metrics, enriched_metrics = load_model_metrics()

        metric_cols = st.columns(3)
        for col, metric in zip(metric_cols, ["Accuracy", "F1 (macro)", "AUC"]):
            delta = round(enriched_metrics[metric] - baseline_metrics[metric], 4)
            col.metric(
                label=metric,
                value=enriched_metrics[metric],
                delta=f"{delta:+.4f} vs baseline",
            )

        st.markdown("---")

        comparison_df = pd.DataFrame({
            "Baseline (S5: degree only)":           baseline_metrics,
            "Graph Enriched (S6: +pagerank +comm)": enriched_metrics,
        }).T

        for m in ["Accuracy", "F1 (macro)", "AUC"]:
            comparison_df[f"Delta {m}"] = (
                comparison_df[m] - baseline_metrics[m]
            ).round(4)

        st.dataframe(comparison_df, use_container_width=True)

    except FileNotFoundError:
        st.error(
            "Metrics files not found. "
            "Run Notebook 04 first to generate "
            "reports/baseline_metrics.json and reports/enriched_metrics.json."
        )

# =========================================================
# TAB 3 — GRAPH ANALYTICS
# =========================================================

with tab_graph:

    st.header("Neo4j Graph Analytics")
    st.markdown("""
    Data comes from the `pagerank` and `community` node properties
    written by `03_graph_analytics.ipynb` via GDS.
    If Neo4j properties are not available, falls back to `graph_features.csv`.
    """)

    left_col, right_col = st.columns(2)

    # ── PageRank ──────────────────────────────────────────
    with left_col:

        st.subheader("Top Orders by PageRank")
        st.caption(
            "Most structurally central orders. "
            "High PageRank = connected to many customers, regions and departments."
        )

        top_n = st.slider(
            "Select Number of Orders",
            min_value=5, max_value=20, value=10,
        )

        pagerank_df = load_top_pagerank_orders(limit=top_n)

        if not pagerank_df.empty:
            fig_pr = px.bar(
                pagerank_df,
                x="order_id",
                y="pagerank",
                color="pagerank",
                color_continuous_scale="Blues",
                title=f"Top {top_n} Orders by PageRank",
                labels={
                    "order_id": "Order ID",
                    "pagerank": "PageRank Score",
                },
            )
            fig_pr.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_pr, use_container_width=True)
            st.dataframe(pagerank_df, use_container_width=True)
        else:
            st.error(
                "No PageRank data found. "
                "Make sure 03_graph_analytics.ipynb ran successfully "
                "AND graph_features.csv exists in data/clean/."
            )

    # ── Community Distribution ────────────────────────────
    with right_col:

        st.subheader("Community Distribution")
        st.caption(
            "Each community is a densely connected subgroup "
            "detected by the Louvain algorithm."
        )

        community_df = load_community_distribution()

        if not community_df.empty:
            fig_comm = px.bar(
                community_df,
                x="community",
                y="order_count",
                color="order_count",
                color_continuous_scale="Oranges",
                title="Order Count per Community (Top 10)",
                labels={
                    "community":   "Community ID",
                    "order_count": "Number of Orders",
                },
            )
            st.plotly_chart(fig_comm, use_container_width=True)
            st.dataframe(community_df, use_container_width=True)
        else:
            st.error(
                "No community data found. "
                "Make sure 03_graph_analytics.ipynb ran successfully "
                "AND graph_features.csv exists in data/clean/."
            )

# =========================================================
# TAB 4 — SEMANTIC SIMILARITY SEARCH
# =========================================================

with tab_similarity:

    st.header("Semantic Product Similarity Search")

    st.markdown("""
    Search for supply chain products using semantic embeddings.
    The dashboard embeds the query, retrieves nearest products from Qdrant,
    shows delivery-risk analytics, and auto-fills the Order Risk Predictor
    tab with real order data for that product.
    """)

    query_text = st.text_input(
        "Enter Search Query",
        placeholder="e.g. sporting goods with shipping delays",
    )

    category_filter = st.text_input(
        "Optional Category Filter",
        placeholder="e.g. Cleats",
    )

    top_k = st.slider("Number of Results", min_value=3, max_value=10, value=5)

    if query_text:

        try:

            similarity_df = perform_similarity_search(
                query_text=query_text,
                top_k=top_k,
                category_filter=category_filter or None,
            )

            if similarity_df.empty:

                st.warning("No similar products found. Run 05_embeddings.ipynb to build the Qdrant index.")

            else:

                st.success(f"Retrieved {len(similarity_df)} similar products.")

                # Convert numeric cols before gradient styling
                similarity_df["similarity_score"]    = pd.to_numeric(similarity_df["similarity_score"],    errors="coerce")
                similarity_df["avg_risk_probability"] = pd.to_numeric(similarity_df["avg_risk_probability"], errors="coerce")

                st.dataframe(
                    similarity_df.style.background_gradient(
                        subset=["similarity_score", "avg_risk_probability"],
                        cmap="RdYlGn_r",
                    ),
                    use_container_width=True,
                )

                st.markdown("---")

                # ── Product selection ──────────────────────────────────
                selected_product = st.selectbox(
                    "Select a Product to Analyse",
                    similarity_df["product_name"].tolist(),
                )

                selected_row = similarity_df[
                    similarity_df["product_name"] == selected_product
                ].iloc[0]

                # ── Look up real order data from merged_df ─────────────
                merged_df = load_merged_df()
                graph_df  = load_graph_features_csv()

                product_orders = merged_df[
                    merged_df["Product Name"].str.lower().str.strip()
                    == selected_product.lower().strip()
                ]

                if not product_orders.empty:
                    rep_order    = product_orders.iloc[0]
                    order_id_str = str(rep_order["Order Id"])
                    graph_row    = graph_df[graph_df["order_id"] == order_id_str]

                    pagerank_val  = float(graph_row["pagerank"].iloc[0])  if not graph_row.empty else 0.0
                    community_val = str(graph_row["community"].iloc[0])   if not graph_row.empty else "0"
                    degree_val    = int(graph_row["degree"].iloc[0])      if not graph_row.empty and "degree"     in graph_row.columns else 3
                    out_deg_val   = int(graph_row["out_degree"].iloc[0])  if not graph_row.empty and "out_degree" in graph_row.columns else 2
                    in_deg_val    = int(graph_row["in_degree"].iloc[0])   if not graph_row.empty and "in_degree"  in graph_row.columns else 1

                    auto_features = {
                        "Sales":               float(rep_order.get("Sales", 100.0)),
                        "Order Item Quantity": int(rep_order.get("Order Item Quantity", 5)),
                        "degree":              degree_val,
                        "out_degree":          out_deg_val,
                        "in_degree":           in_deg_val,
                        "pagerank":            pagerank_val,
                        "Shipping Mode":       str(rep_order.get("Shipping Mode", "Standard Class")),
                        "Order Region":        str(rep_order.get("Order Region",   "Western Europe")),
                        "Order Status":        str(rep_order.get("Order Status",   "COMPLETE")),
                        "community":           community_val,
                        "product_name":        selected_product,
                        "similarity_score":    float(selected_row["similarity_score"]),
                        "avg_delay_days":      float(selected_row["avg_delay_days"])      if pd.notna(selected_row["avg_delay_days"])      else 0.0,
                        "avg_risk_probability":float(selected_row["avg_risk_probability"]) if pd.notna(selected_row["avg_risk_probability"]) else 0.0,
                    }
                else:
                    auto_features = {
                        "Sales":               100.0,
                        "Order Item Quantity": 5,
                        "degree":              3,
                        "out_degree":          2,
                        "in_degree":           1,
                        "pagerank":            0.5,
                        "Shipping Mode":       "Standard Class",
                        "Order Region":        "Western Europe",
                        "Order Status":        "COMPLETE",
                        "community":           "0",
                        "product_name":        selected_product,
                        "similarity_score":    float(selected_row["similarity_score"]),
                        "avg_delay_days":      float(selected_row["avg_delay_days"])      if pd.notna(selected_row["avg_delay_days"])      else 0.0,
                        "avg_risk_probability":float(selected_row["avg_risk_probability"]) if pd.notna(selected_row["avg_risk_probability"]) else 0.0,
                    }

                # Store for Tab 5
                st.session_state["auto_features"] = auto_features

                # ── Product analytics metrics ──────────────────────────
                st.subheader("Selected Product Analytics")

                col1, col2, col3 = st.columns(3)
                col1.metric("Similarity Score",    round(float(selected_row["similarity_score"]), 4))
                col2.metric("Avg Delay Days",      round(float(selected_row["avg_delay_days"])      if pd.notna(selected_row["avg_delay_days"])      else 0.0, 2))
                col3.metric("Avg Risk Probability",round(float(selected_row["avg_risk_probability"]) if pd.notna(selected_row["avg_risk_probability"]) else 0.0, 4))

                # ── Auto prediction with REAL order data ───────────────
                st.markdown("---")
                st.subheader("Automatic Delivery Risk Prediction")
                st.caption(
                    f"Using real order data for: **{selected_product}**  "
                    f"· Shipping Mode: {auto_features['Shipping Mode']}  "
                    f"· Region: {auto_features['Order Region']}  "
                    f"· PageRank: {auto_features['pagerank']:.6f}"
                )

                feature_dict = {
                    "Sales":               auto_features["Sales"],
                    "Order Item Quantity": auto_features["Order Item Quantity"],
                    "degree":              auto_features["degree"],
                    "out_degree":          auto_features["out_degree"],
                    "in_degree":           auto_features["in_degree"],
                    "pagerank":            auto_features["pagerank"],
                    "Shipping Mode":       auto_features["Shipping Mode"],
                    "Order Region":        auto_features["Order Region"],
                    "Order Status":        auto_features["Order Status"],
                    "community":           auto_features["community"],
                }

                prediction, probability = predict_order_risk(feature_dict)

                pred_col1, pred_col2 = st.columns(2)
                pred_col1.metric(
                    "Prediction",
                    "🚨 LATE DELIVERY" if prediction == 1 else "✅ ON-TIME DELIVERY",
                )
                pred_col2.metric("Late Delivery Probability", f"{probability:.1%}")

                if probability > 0.7:
                    st.error("High delivery-risk product detected.")
                elif probability > 0.4:
                    st.warning("Moderate delivery risk detected.")
                else:
                    st.success("Low delivery risk detected.")

                st.info("➡️ Switch to the **Order Risk Predictor** tab to adjust values and re-run the prediction.")

        except Exception as error:
            st.error(f"Similarity search failed: {error}")


# =========================================================
# TAB 5 — ORDER RISK PREDICTOR
# =========================================================

with tab_prediction:

    st.header("Late Delivery Risk Predictor")

    st.markdown("""
    Enter order details. The graph-enriched pipeline
    (`enriched_pipeline.joblib`) predicts whether this order
    is likely to arrive late and returns the probability.
    """)

    # =========================================
    # AUTO LOAD FEATURES FROM TAB 4
    # =========================================

    auto_data = st.session_state.get(
        "auto_features",
        None
    )

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        # =====================================
        # COLUMN 1
        # =====================================

        with col1:

            sales = st.number_input(
                "Sales",
                min_value=0.0,
                value=(
                    auto_data["Sales"]
                    if auto_data else 100.0
                ),
            )

            quantity = st.number_input(
                "Order Quantity",
                min_value=1,
                value=(
                    auto_data["Order Item Quantity"]
                    if auto_data else 5
                ),
            )

            degree = st.number_input(
                "Degree",
                min_value=0,
                value=(
                    auto_data["degree"]
                    if auto_data else 3
                ),
            )

        # =====================================
        # COLUMN 2
        # =====================================

        with col2:

            out_degree = st.number_input(
                "Out Degree",
                min_value=0,
                value=(
                    auto_data["out_degree"]
                    if auto_data else 2
                ),
            )

            in_degree = st.number_input(
                "In Degree",
                min_value=0,
                value=(
                    auto_data["in_degree"]
                    if auto_data else 1
                ),
            )

            pagerank = st.number_input(
                "PageRank",
                min_value=0.0,
                value=(
                    auto_data["pagerank"]
                    if auto_data else 0.5
                ),
                format="%.4f",
            )

        # =====================================
        # COLUMN 3
        # =====================================

        with col3:

            shipping_options = [
                "Standard Class",
                "Second Class",
                "First Class",
                "Same Day",
            ]

            shipping_mode = st.selectbox(
                "Shipping Mode",
                shipping_options,
                index=(
                    shipping_options.index(
                        auto_data["Shipping Mode"]
                    )
                    if auto_data else 0
                ),
            )

            order_region = st.text_input(
                "Order Region",
                (
                    auto_data["Order Region"]
                    if auto_data else "Western Europe"
                ),
            )

            order_status = st.text_input(
                "Order Status",
                (
                    auto_data["Order Status"]
                    if auto_data else "COMPLETE"
                ),
            )

            community = st.text_input(
                "Community",
                (
                    auto_data["community"]
                    if auto_data else "0"
                ),
            )

        # =====================================
        # SUBMIT BUTTON
        # =====================================

        submit_button = st.form_submit_button(
            "Predict Risk"
        )

    # =========================================
    # PREDICTION LOGIC
    # =========================================

    if submit_button:

        feature_dict = {

            "Sales":
                sales,

            "Order Item Quantity":
                quantity,

            "degree":
                degree,

            "out_degree":
                out_degree,

            "in_degree":
                in_degree,

            "pagerank":
                pagerank,

            "Shipping Mode":
                shipping_mode,

            "Order Region":
                order_region,

            "Order Status":
                order_status,

            "community":
                community,
        }

        try:

            prediction, probability = (
                predict_order_risk(
                    feature_dict
                )
            )

            result_col1, result_col2 = (
                st.columns(2)
            )

            result_col1.metric(
                "Prediction",
                (
                    "🚨 LATE DELIVERY"
                    if prediction == 1
                    else "✅ ON-TIME DELIVERY"
                ),
            )

            result_col2.metric(
                "Late Delivery Probability",
                f"{probability:.1%}",
            )

            # =================================
            # RISK ALERTS
            # =================================

            if probability > 0.7:

                st.error(
                    "High risk — immediate intervention recommended."
                )

            elif probability > 0.4:

                st.warning(
                    "Moderate risk — monitor this order carefully."
                )

            else:

                st.success(
                    "Low delivery risk detected."
                )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )
