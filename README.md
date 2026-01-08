

## Project Overview

End-to-end data analytics project demonstrating:

| Component | Technologies |
|-----------|-------------|
| **Data Pipeline** | Python, Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Isolation Forest, Random Forest) |
| **Database** | SQL Queries |
| **Visualization** | Streamlit, Plotly |
| **Web Dashboard** | HTML/CSS/JS, GitHub Pages |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data & Train Model

```bash
python main.py
```

This generates:
- 5,000 customers
- 12,000 insurance claims
- ML fraud risk predictions
- CSV exports in `data/`

### 3. Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The Streamlit dashboard includes:
- **Executive Overview**: KPI tracking and monthly volume trends.
- **Fraud Detection**: Deep dive into ML model risk scores and anomaly detection.
- **Customer Insights**: Demographics, behavioral analytics, and premium tier analysis.
- **Data Explorer**: Interactive search for individual claims.

## Project Structure

```
├── main.py                  # Full pipeline runner
├── streamlit_app.py         # Dashboard application
├── requirements.txt         # Dependencies
├── src/
│   ├── data_generator.py    # Synthetic data generation
│   ├── fraud_detection.py   # ML model (Isolation Forest + Random Forest)
│   └── export_web_data.py   # JSON export for web
├── sql/                     # SQL analytical scripts
│   ├── portfolio_analysis.sql
│   └── fraud_queries.sql
├── data/                    # Generated CSVs (ignored in git)
└── docs/                    # GitHub Pages website
    ├── index.html
    ├── styles.css
    ├── app.js
    └── data/dashboard.json
```

## Machine Learning Model

**Ensemble approach combining:**
- **Isolation Forest** - Unsupervised anomaly detection
- **Random Forest** - Supervised classification

**Results:**
- ROC-AUC Score: **1.0**
- 6.7% fraud detection rate
- Top features: `amount_vs_avg_ratio`, `documentation_score`, `late_reporting_days`

## GitHub Pages Deployment

1. Push to GitHub
2. Go to **Settings > Pages**
3. Set source to `main` branch, `/docs` folder
4. Your site will be live at `https://username.github.io/repo-name/`

## License

MIT License

---

*Built for TD Insurance Data Analytics Internship Application*
