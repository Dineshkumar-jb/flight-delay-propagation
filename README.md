# Flight Delay Propagation Analysis using Sequential and Spatial Data Mining Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Leaflet.js](https://img.shields.io/badge/Leaflet-v1.9.4-199900?style=flat&logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Executive Summary
Commercial aircraft operate in continuous sequential multi-leg itineraries (e.g., **Leg 1: LAS ➔ SEA**, **Leg 2: SEA ➔ SLC**, **Leg 3: SLC ➔ JFK**). When an initial disruption occurs at an origin hub airport, tight turnaround buffer schedules cause arrival delays to propagate downstream to subsequent flight legs—a phenomenon known as **cascade or knock-on delay**.

Standard single-flight predictive models evaluate flights as isolated, independent events, failing to capture:
1. **Sequential Structure**: Accumulative delay along multi-leg aircraft chains per tail number.
2. **Spatial Structure**: Geographic clustering of congested hub regions and weather zones.

This repository implements an end-to-end Python mining engine that integrates **PrefixSpan pattern growth**, **spherical Haversine ST-DBSCAN spatial clustering**, **NetworkX hub centrality**, and **Spatio-Sequential Machine Learning classifiers (SVM RBF, Random Forest, Gradient Boosting)** to analyze U.S. Department of Transportation Bureau of Transportation Statistics (BTS) flight telemetry.

---

## 🎓 Academic Metadata
- **Course**: CSE3068 - Sequential and Spatial Data Mining (DA-2 Deliverable)
- **Slot**: D1 + TD1 + TDD1
- **Student Name**: DINESH KUMAR J B
- **Register No**: 23MIA1126
- **Institution**: School of Computer Science and Engineering (SCOPE), VIT Chennai
- **Dataset Source**: [U.S. DOT BTS Airline On-Time Performance Telemetry](https://www.kaggle.com/datasets/daryaheyko/airline-on-time-statistics-and-delay-causes-bts)

---

## 📐 System Architecture & Data Flow Diagram (DFD)

### 1. System Architecture
```mermaid
graph TD
    classDef titleStyle fill:#1e293b,stroke:#0f172a,color:#ffffff,font-weight:bold;
    classDef sourceStyle fill:#e0f2fe,stroke:#0284c7,color:#0369a1,font-weight:bold;
    classDef prepStyle fill:#ffffff,stroke:#475569,color:#1e293b;
    classDef spmStyle fill:#ffffff,stroke:#ea580c,color:#9a3412;
    classDef sdmStyle fill:#ffffff,stroke:#16a34a,color:#166534;
    classDef integStyle fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,font-weight:bold;
    classDef visStyle fill:#ffffff,stroke:#9333ea,color:#6b21a8;
    classDef evalStyle fill:#ffffff,stroke:#ca8a04,color:#854d0e;
    classDef sinkStyle fill:#f1f5f9,stroke:#334155,color:#0f172a,font-weight:bold;

    Title["SYSTEM ARCHITECTURE: FLIGHT DELAY PROPAGATION ANALYSIS<br/>DINESH KUMAR J B (23MIA1126)"]:::titleStyle
    Source["Raw Flight Operations Data<br/>(US DOT BTS Dataset)"]:::sourceStyle

    subgraph DataPrep ["Data Preparation Layer"]
        P1["1.1 Data Cleaning & Imputation"]:::prepStyle
        P2["1.2 Flight Chain Construction &lt;Airport, Delay&gt;"]:::prepStyle
        P3["1.3 Airport Coordinate Attachment (Haversine)"]:::prepStyle
    end

    subgraph SPM ["Sequential Pattern Mining (SPM)"]
        SPM1["Edit Distance Alignment"]:::spmStyle
        SPM2["GSP Algorithm (Apriori BFS)"]:::spmStyle
        SPM3["PrefixSpan (Pattern Growth DFS)"]:::spmStyle
    end

    subgraph SDM ["Spatial Data Mining (SDM)"]
        SDM1["Spatial Association Mining"]:::sdmStyle
        SDM2["ST-DBSCAN Clustering"]:::sdmStyle
    end

    Integ["Spatio-Sequential Integration Core<br/>Merge Sequential Trajectories with Spatial Clusters"]:::integStyle

    subgraph Vis ["Visualization Layer"]
        V1["Interactive Delay Map (Leaflet)"]:::visStyle
        V2["Airport Network Graph (NetworkX)"]:::visStyle
    end

    subgraph Eval ["Evaluation Layer"]
        E1["SVM / RF / GB Classifiers"]:::evalStyle
        E2["Silhouette Score"]:::evalStyle
        E3["Support / Confidence Rules"]:::evalStyle
    end

    Sink["Final Output: Delay Patterns &amp; Spatial Insights"]:::sinkStyle

    Title --> Source
    Source --> DataPrep
    P1 --> P2
    P2 --> P3
    P3 --> SPM
    P3 --> SDM
    SPM --> Integ
    SDM --> Integ
    Integ --> Vis
    Integ --> Eval
    Vis --> Sink
    Eval --> Sink
```

### 2. Level-1 Data Flow Diagram (DFD Notation)
```mermaid
graph TD
    classDef entityStyle fill:#ffffff,stroke:#000000,stroke-width:2.5px,color:#000000,font-weight:bold;
    classDef processStyle fill:#ffffff,stroke:#000000,stroke-width:2px,color:#000000,font-weight:bold;
    classDef processCoreStyle fill:#e6e6e6,stroke:#000000,stroke-width:2.5px,color:#000000,font-weight:bold;
    classDef storeStyle fill:#f8fafc,stroke:#000000,stroke-width:2px,color:#000000,font-weight:bold;

    %% External Entities (Rectangles with bold outlines)
    E1["External Entity [E1]:<br/>U.S. DOT BTS Flight Data Repository"]:::entityStyle
    E2["External Entity [E2]:<br/>Airline Ops & FAA Controller Dashboard"]:::entityStyle

    %% Processes (Circular / Bubble Nodes with Process Numbers)
    P1(("1.0<br/>Ingest & Preprocess<br/>Flight Telemetry")):::processStyle
    P2(("2.0<br/>Mine Sequential<br/>Delay Patterns")):::processStyle
    P3(("3.0<br/>Mine Spatial<br/>Airport Clusters")):::processStyle
    P4(("4.0<br/>Integrate Spatio-Seq<br/>Graph & Train ML")):::processCoreStyle
    P5(("5.0<br/>Render Interactive<br/>Map & Dashboard")):::processStyle

    %% Data Stores (Cylinders)
    D1[("(D1) Flight Sequence Repository")]:::storeStyle
    D2[("(D2) Frequent Sequential Pattern Base")]:::storeStyle
    D3[("(D3) Spatial Cluster & Rule Index")]:::storeStyle
    D4[("(D4) Propagation Graph & ML Model Store")]:::storeStyle

    %% Labeled Data Flows (Directional Connectors with Data Payloads)
    E1 -- "Raw Flight Telemetry & Coordinates" --> P1
    P1 -- "Cleaned Flight Sequences" --> D1
    D1 -- "Tail-Number Flight Chains" --> P2
    D1 -- "Airport Coordinates & Distances" --> P3
    P2 -- "Frequent s-patterns & Rules" --> D2
    P3 -- "Spatial Hotspots & Clusters" --> D3
    D2 -- "Sequential Delay Trajectories" --> P4
    D3 -- "Geodesic Cluster Geometries" --> P4
    P4 -- "Trained ML Models & Risk Scores" --> D4
    D4 -- "Propagation Graph & Centrality Metrics" --> P5
    P5 -- "Interactive Delay Map & Alerts" --> E2
```

---

## 📊 Visual Analytics & Graphical Figures

| Figure # | Analytics Description | Chart Category | Image Artifact |
| :---: | :--- | :--- | :---: |
| **Figure 1** | Top 10 Delayed Origin Airports | Lollipop Ranking Chart | ![Fig 1](output_images/plot_1_lollipop_origin_delays.png) |
| **Figure 2** | Arrival Delay Density PDF | Shaded KDE PDF Curve | ![Fig 2](output_images/plot_2_kde_delay_distribution.png) |
| **Figure 3** | Multi-Leg Itinerary Share | Donut Ring Chart | ![Fig 3](output_images/plot_3_donut_itinerary_legs.png) |
| **Figure 4** | DBSCAN Radius Epsilon Sensitivity | Dual Step Plot | ![Fig 4](output_images/plot_4_dual_step_dbscan_sensitivity.png) |
| **Figure 5** | PrefixSpan Frequency Sensitivity | Vertical Pin Chart | ![Fig 5](output_images/plot_5_vertical_pin_prefixspan.png) |
| **Figure 6** | Spatio-Sequential Flight Network | Geographic Coordinate Network Map | ![Fig 6](output_images/plot_6_geographic_network_graph.png) |
| **Figure 7** | Arrival Delay Density per Spatial Cluster | Violin Density Plot | ![Fig 7](output_images/plot_7_violin_cluster_delays.png) |
| **Figure 8** | Geographic Delay Density Hotspots | Hexbin Binned Spatial Matrix | ![Fig 8](output_images/plot_8_hexbin_delay_map.png) |
| **Figure 9** | Top Mined Delay Propagation Chains | Diverging Color Ranking Chart | ![Fig 9](output_images/plot_9_diverging_trajectory_ranking.png) |
| **Figure 10** | Spatio-Sequential Disruption Heatmap | Annotated Matrix Heatmap | ![Fig 10](output_images/plot_10_heatmap_disruption_matrix.png) |
| **Figure 11** | Multi-Metric Classifier Comparison | Polar Radar / Spider Web Chart | ![Fig 11](output_images/plot_11_radar_model_evaluation.png) |

---

## 🔬 Benchmark Evaluation Results

Predictive models evaluate downstream delay cascade risk using spatio-sequential feature vectors $X = [\text{leg\_idx}, \text{dist\_km}, \text{sched\_dep}, \text{origin\_cluster}, \text{dest\_cluster}, \text{upstream\_delay}, \text{carryover\_delay}, \text{turnaround\_risk}]$.

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proposed Spatio-Sequential SVM (RBF)** | **0.7950** | **0.8667** | **0.2500** | **0.3881** | **0.7539** |
| **Proposed Spatio-Sequential Random Forest** | 0.7700 | 0.6667 | 0.2308 | 0.3429 | 0.7143 |
| **Proposed Spatio-Sequential Gradient Boosting** | 0.7150 | 0.4138 | 0.2308 | 0.2963 | 0.6344 |
| **Baseline Single-Flight Logistic Regression** | 0.7400 | 0.5000 | 0.0192 | 0.0370 | 0.5032 |

---

## ⚡ Quickstart Guide

### 1. Requirements & Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/Dineshkumar-jb/flight-delay-propagation.git
cd flight-delay-propagation
pip install pandas numpy scikit-learn matplotlib seaborn networkx balltree
```

### 2. Execute Master Mining Pipeline
Run the main script to process telemetry, mine delay sequences, construct spatial DBSCAN clusters, train predictive classifiers, and update all visual plots:
```bash
python3 da2_flight_delay_propagation.py
```

### 3. Open Interactive 60 FPS Leaflet Dashboard
Open `flight_delay_propagation_map.html` in any web browser to explore Great-Circle geodesic flight arcs, airport hub search dock, and live delay telemetry risk overlays:
```bash
# On Linux
xdg-open flight_delay_propagation_map.html
```

---

## 📚 Domain References
1. M. Pyrgiotis, K. M. Malone, and A. Odoni, "Modelling Delay Propagation within an Airport Network," *Transportation Research Part C: Emerging Technologies*, vol. 27, pp. 60–75, 2013.
2. S. Wandelt and X. Sun, "Spatial and Temporal Evolution of Delay Propagation in U.S. Domestic Air Transportation," *IEEE Transactions on Intelligent Transportation Systems*, vol. 16, no. 6, pp. 3185–3194, 2015.
3. A. Sahin, P. V. Hien, and E. E. Oztekin, "A Spatio-Temporal Approach for Modeling Flight Delay Cascades in Airport Networks," *Journal of Air Transport Management*, vol. 92, p. 102021, 2021.
4. N. Xu, M. G. Ball, and S. E. Zou, "Flight Delay Propagation Analysis in Multi-Airport Networks," *Transportation Research Part C: Emerging Technologies*, vol. 18, no. 5, pp. 673–685, 2010.
5. U.S. Department of Transportation Bureau of Transportation Statistics (BTS), "Airline On-Time Performance Flight Delays Dataset," Federal Aviation Administration (FAA), Washington, D.C., 2024.

---
*Developed by **DINESH KUMAR J B** (Reg No: 23MIA1126), SCOPE, VIT Chennai.*
