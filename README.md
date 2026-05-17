📦 Supply Chain Intelligence Platform
An end-to-end graph-enhanced supply chain analytics platform built using Neo4j, Graph Data Science (GDS), Machine Learning, SentenceTransformer embeddings, Qdrant vector search, and Streamlit.
The project combines graph analytics, machine learning, and semantic similarity search to improve supply chain visibility and delivery-risk prediction.
________________________________________
🎯 Business Question
Which supply chain factors—across products, regions, and shipping modes—are driving late deliveries, and what predictive signals can be used to reduce delays? 
________________________________________
🚀 Key Features
•	Neo4j supply chain knowledge graph
•	Graph analytics using PageRank and Louvain community detection
•	Baseline vs graph-enriched ML pipelines
•	SentenceTransformer embeddings
•	Qdrant vector similarity search
•	Streamlit business intelligence dashboard
•	Dockerized infrastructure
________________________________________
🧱 Tech Stack
Layer	Technology
Data Processing	pandas, DuckDB
Graph Database	Neo4j
Graph Analytics	Neo4j GDS
Machine Learning	scikit-learn
Embeddings	SentenceTransformers
Vector Database	Qdrant
Dashboard	Streamlit
Visualization	Plotly
Containerization	Docker
________________________________________
🏗️ System Architecture
Raw CSV Data
      ↓
DuckDB ETL
      ↓
Neo4j Graph Construction
      ↓
Graph Analytics (GDS)
      ↓
Feature Enrichment
      ↓
Machine Learning
      ↓
Embeddings + Qdrant
      ↓
Streamlit Dashboard
________________________________________
# 📂 Project Structure
BigData_and_BI_CapstoneProject/
- docker-compose.yml
- README.md
- requirements.txt
- .gitignore

- data/
  - raw/
  - clean/
  - download.sh

- notebooks/
  - 01_etl.ipynb
  - 02_graph_load.ipynb
  - 03_graph_analytics.ipynb
  - 03_graph_analytics_1.ipynb
  - 3_graph_analytics.ipynb
  - 04_ml.ipynb
  - 4_ml.ipynb
  - 05_embeddings.ipynb
  - 5_embeddings.ipynb

- app/
  - streamlit_app.py

- docs/
  - Project_Proposal.docx

- images/
  - graph_fitness.png

- models/

- reports/
  - baseline_metrics.json
  - enriched_metrics.json
________________________________________
📊 Dataset
DataCo Smart Supply Chain Dataset
Dataset Source:
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
________________________________________
⚙️ Setup Instructions
1. Clone Repository
git clone <repository-url>
cd BigData_and_BI_CapstoneProject

2. Create Virtual Environment
Windows
python -m venv env
env\Scripts\activate
Mac/Linux
python3 -m venv env
source env/bin/activate

3. Install Dependencies
pip install -r requirements.txt

________________________________________
📥 Dataset Setup
Run:
bash data/download.sh
Then manually place downloaded CSV files inside:
data/raw/

________________________________________

🔐 Environment Variables
Create a .env file in the project root.
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
________________________________________
🐳 Run Infrastructure
Start Neo4j and Qdrant:
docker-compose up -d
Services:
Service	URL
Neo4j Browser	http://localhost:7474
Qdrant	http://localhost:6333
________________________________________
📘 Notebook Execution Order
Notebook	Purpose
01_etl.ipynb	Data cleaning and DuckDB ETL
02_graph_load.ipynb	Load graph into Neo4j
03_graph_analytics.ipynb	Run PageRank and Louvain
04_ml.ipynb	Baseline and graph-enriched ML pipelines
05_embeddings.ipynb	Embeddings and Qdrant similarity search
________________________________________
🤖 Machine Learning Pipeline
Baseline Model
Traditional ML features:
•	Sales
•	Quantity
•	Shipping Mode
•	Order Status
•	Order Region
Graph-Enriched Model
Additional graph features:
•	degree
•	out_degree
•	in_degree
•	pagerank
•	community
The graph-enriched model improves prediction quality by capturing hidden network relationships.
________________________________________
🔗 Graph Analytics
Neo4j Graph Data Science algorithms used:
•	PageRank
•	Louvain Community Detection
These algorithms identify:
•	structurally important nodes
•	network influence
•	hidden communities
•	connectivity patterns
________________________________________
🔍 Semantic Similarity Search
The project uses:
•	SentenceTransformer embeddings
•	Cosine similarity
•	Qdrant vector database
Users can search semantically similar products directly from the Streamlit dashboard.
________________________________________
📈 Streamlit Dashboard Features
•	Business KPI monitoring
•	Baseline vs enriched model comparison
•	Neo4j graph analytics visualization
•	Community distribution analysis
•	Semantic product similarity search
•	Real-time late-delivery risk prediction
Run dashboard:
streamlit run app/streamlit_app.py
________________________________________
📌 Future Improvements
•	Real-time streaming predictions
•	Graph embeddings
•	Kubernetes deployment
•	LLM-powered supply chain assistant
•	Recommendation engine
________________________________________
👨‍💻 Team Members
Arya Tamhane
Rudra Pingale
Juliee Patil
Harsha Vardhan Pandiri
________________________________________
