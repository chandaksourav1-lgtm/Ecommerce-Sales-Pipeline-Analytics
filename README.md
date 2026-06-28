# E-Commerce Startup — Sales Pipeline Analytics Dashboard

## Business Problem
A newly launched e-commerce startup suspects that deal closures in its sales pipeline are being influenced by biased or inefficient factors — potentially driven by acquisition channel, product category, customer segment, sales rep performance, or regional differences rather than genuine market demand.

## Objectives
1. **Descriptive Analysis** — Cross-tabulation and frequency analysis segmented by Pipeline Stage
2. **Diagnostic Analysis** — Bias detection across channels, regions, reps, and customer segments (with Chi-Square, ANOVA, and T-tests)
3. **Classification Models** — KNN, Decision Tree, Random Forest, and Gradient Boosted Trees with SMOTE oversampling
4. **Model Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC Curves, and Confusion Matrices
5. **Key Findings** — Actionable insights and recommendations

## Dataset
- 1,000 synthetic sales pipeline records
- 27 features after cleaning and transformation
- Target variable: `Deal_Won` (1 = Closed Won, 0 = Not Won)

## Project Structure
```
ecommerce_project/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── src/
│   └── analysis.py         # Standalone analysis script
├── data/
│   ├── ecommerce_sales_pipeline.xlsx    # Raw dataset
│   └── ecommerce_cleaned_data.xlsx      # Cleaned dataset
└── plots/
    ├── 01_accuracy_comparison.png
    ├── 02_precision_recall_f1.png
    ├── 03_roc_curves.png
    ├── 04_confusion_matrices.png
    ├── 05_feature_importance.png
    ├── 06_channel_conversion.png
    ├── 07_region_winrate.png
    ├── 08_rep_performance.png
    ├── 09_segment_analysis.png
    └── 10_correlation_heatmap.png
```

## How to Run

### Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Click Deploy

## Tools & Libraries
- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn, Imbalanced-learn (SMOTE)
- Streamlit (interactive dashboard)
- SciPy (statistical tests)
