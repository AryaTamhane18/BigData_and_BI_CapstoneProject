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

@st.cache_data(ttl=300)
def load_prediction_data():
    return pd.read_csv(PREDICTIONS_FILE)

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
    Type a supply chain query. The dashboard embeds it with
    `all-MiniLM-L6-v2` (same model as `05_embeddings.ipynb`),
    queries Qdrant for the nearest products, and shows their
    delivery risk alongside the similarity score.
    This is the **R + A** (Retrieve + Augment) step of the RAG architecture.
    """)

    query_text      = st.text_input(
        "Enter Search Query",
        placeholder="e.g. sporting goods with high shipping delays",
    )
    category_filter = st.text_input(
        "Optional: Filter by Category",
        placeholder="e.g. Cleats",
    )
    top_k = st.slider("Number of Results", min_value=3, max_value=10, value=5)

    if query_text:
        with st.spinner("Embedding query and searching Qdrant..."):
            try:
                similarity_df = perform_similarity_search(
                    query_text=query_text,
                    top_k=top_k,
                    category_filter=category_filter or None,
                )
                if similarity_df.empty:
                    st.warning(
                        "No results found. "
                        "Run 05_embeddings.ipynb to build the Qdrant index."
                    )
                else:
                    st.success(
                        f"Top {len(similarity_df)} products similar to: "
                        f"*{query_text}*"
                    )
                    
                    similarity_df["similarity_score"]    = pd.to_numeric(similarity_df["similarity_score"],    errors="coerce")
                    similarity_df["avg_risk_probability"] = pd.to_numeric(similarity_df["avg_risk_probability"], errors="coerce")

                    st.dataframe(
                        similarity_df.style.background_gradient(
                            subset=["similarity_score", "avg_risk_probability"],
                            cmap="RdYlGn_r",
                        ),
                        use_container_width=True,
                    )
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

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            sales    = st.number_input("Sales",          min_value=0.0, value=100.0)
            quantity = st.number_input("Order Quantity",  min_value=1,   value=5)
            degree   = st.number_input("Degree",          min_value=0,   value=3)

        with col2:
            out_degree = st.number_input("Out Degree", min_value=0,   value=2)
            in_degree  = st.number_input("In Degree",  min_value=0,   value=1)
            pagerank   = st.number_input("PageRank",   min_value=0.0, value=0.5, format="%.4f")

        with col3:
            shipping_mode = st.selectbox(
                "Shipping Mode",
                ["Standard Class", "Second Class", "First Class", "Same Day"],
            )
            order_region = st.text_input("Order Region", "Western Europe")
            order_status = st.text_input("Order Status", "COMPLETE")
            community    = st.text_input("Community",    "0")

        submit_button = st.form_submit_button("Predict Risk")

    if submit_button:

        feature_dict = {
            "Sales":               sales,
            "Order Item Quantity": quantity,
            "degree":              degree,
            "out_degree":          out_degree,
            "in_degree":           in_degree,
            "pagerank":            pagerank,
            "Shipping Mode":       shipping_mode,
            "Order Region":        order_region,
            "Order Status":        order_status,
            "community":           community,
        }

        try:
            prediction, probability = predict_order_risk(feature_dict)

            result_col1, result_col2 = st.columns(2)
            result_col1.metric(
                "Prediction",
                "🚨 LATE DELIVERY" if prediction == 1 else "✅ ON-TIME DELIVERY",
            )
            result_col2.metric(
                "Late Delivery Probability",
                f"{probability:.1%}",
            )

            if probability > 0.7:
                st.error("High risk — immediate intervention recommended.")
            elif probability > 0.4:
                st.warning("Moderate risk — monitor this order carefully.")
            else:
                st.success("Low delivery risk detected.")

        except Exception as error:
            st.error(f"Prediction failed: {error}")
