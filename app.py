import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="E-Commerce Sales Pipeline Analytics", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: bold; color: #1F4E79; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem;}
    .metric-card {background: linear-gradient(135deg, #1F4E79, #2E75B6); padding: 1.2rem;
                  border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0;}
    .metric-value {font-size: 1.8rem; font-weight: bold;}
    .metric-label {font-size: 0.85rem; opacity: 0.9;}
    .finding-box {background: #FFF3CD; border-left: 4px solid #FFC107; padding: 1rem;
                  border-radius: 8px; margin: 0.5rem 0;}
    .bias-alert {background: #F8D7DA; border-left: 4px solid #DC3545; padding: 1rem;
                 border-radius: 8px; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_excel("data/ecommerce_sales_pipeline.xlsx", sheet_name="Sales Pipeline Data")
    df['Lead_Date'] = pd.to_datetime(df['Lead_Date'])
    df['Close_Date'] = pd.to_datetime(df['Close_Date'])
    cat_cols = ['Pipeline_Stage', 'Channel', 'Region', 'Customer_Segment', 'Product_Category', 'Sales_Rep']
    for col in cat_cols:
        df[col] = df[col].str.strip().str.title()
    df['CSAT_Available'] = df['CSAT_Score'].notnull().astype(int)
    df['Lead_Month'] = df['Lead_Date'].dt.month
    df['Lead_Quarter'] = df['Lead_Date'].dt.quarter
    df['Lead_DayOfWeek'] = df['Lead_Date'].dt.day_name()
    df['Pipeline_Duration_Days'] = (df['Close_Date'] - df['Lead_Date']).dt.days
    df['Revenue_Per_Unit'] = (df['Revenue'] / df['Quantity']).round(2)
    df['Discount_Band'] = pd.cut(df['Discount_Pct'],
        bins=[-0.01, 0.05, 0.10, 0.20, 1.0],
        labels=['No Discount (0-5%)', 'Low (5-10%)', 'Medium (10-20%)', 'High (>20%)'])
    df['Revenue_Band'] = pd.cut(df['Revenue'],
        bins=[0, 200, 500, 1000, 999999],
        labels=['Low (<200)', 'Medium (200-500)', 'High (500-1K)', 'Premium (>1K)'])
    df['High_Value_Deal'] = (df['Revenue'] >= df['Revenue'].quantile(0.75)).astype(int)
    return df


df = load_data()

st.markdown('<div class="main-header">📊 E-Commerce Startup — Sales Pipeline Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Bias Detection & Predictive Modelling Dashboard</div>', unsafe_allow_html=True)

# ─── SIDEBAR ───
st.sidebar.title("🔧 Filters")
selected_channels = st.sidebar.multiselect("Channel", df['Channel'].unique(), default=df['Channel'].unique())
selected_regions = st.sidebar.multiselect("Region", df['Region'].unique(), default=df['Region'].unique())
selected_segments = st.sidebar.multiselect("Segment", df['Customer_Segment'].unique(), default=df['Customer_Segment'].unique())

df_filtered = df[
    (df['Channel'].isin(selected_channels)) &
    (df['Region'].isin(selected_regions)) &
    (df['Customer_Segment'].isin(selected_segments))
]

# ─── TABS ───
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Descriptive Analysis", "🔍 Diagnostic — Bias Detection",
    "🤖 ML Models", "📈 Model Evaluation", "💡 Findings"
])

# ═══════════════════════════════════════
# TAB 1: DESCRIPTIVE ANALYSIS
# ═══════════════════════════════════════
with tab1:
    st.header("1️⃣ Descriptive Analysis")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Leads", f"{len(df_filtered):,}")
    c2.metric("Total Revenue", f"${df_filtered['Revenue'].sum():,.0f}")
    c3.metric("Closed Won", f"{df_filtered['Deal_Won'].sum()}")
    c4.metric("Win Rate", f"{df_filtered['Deal_Won'].mean()*100:.1f}%")
    c5.metric("Avg Deal Value", f"${df_filtered['Revenue'].mean():,.0f}")
    c6.metric("Avg Days in Pipeline", f"{df_filtered['Days_In_Pipeline'].mean():.0f}")

    st.subheader("Pipeline Stage Distribution")
    col1, col2 = st.columns(2)
    with col1:
        stage_counts = df_filtered['Pipeline_Stage'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#1F4E79', '#2E75B6', '#4A90D9', '#6BAED6', '#9ECAE1', '#2CA02C', '#D62728']
        stage_counts.plot(kind='bar', ax=ax, color=colors[:len(stage_counts)])
        ax.set_title('Leads by Pipeline Stage', fontweight='bold', fontsize=13)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        stage_rev = df_filtered.groupby('Pipeline_Stage')['Revenue'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        stage_rev.plot(kind='bar', ax=ax, color='#2E75B6')
        ax.set_title('Avg Revenue by Pipeline Stage', fontweight='bold', fontsize=13)
        ax.set_ylabel('Revenue ($)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Cross-Tabulation: Pipeline Stage × Key Dimensions")
    cross_dim = st.selectbox("Select dimension:", ['Channel', 'Customer_Segment', 'Region', 'Product_Category'])
    ct = pd.crosstab(df_filtered['Pipeline_Stage'], df_filtered[cross_dim], margins=True)
    st.dataframe(ct, use_container_width=True)

    ct_pct = pd.crosstab(df_filtered['Pipeline_Stage'], df_filtered[cross_dim], normalize='index') * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    ct_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Blues')
    ax.set_title(f'Pipeline Stage × {cross_dim} (% Distribution)', fontweight='bold')
    ax.set_ylabel('Percentage')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Summary Statistics by Pipeline Stage")
    summary = df_filtered.groupby('Pipeline_Stage').agg(
        Count=('Lead_ID', 'count'), Avg_Revenue=('Revenue', 'mean'),
        Median_Revenue=('Revenue', 'median'), Avg_Discount=('Discount_Pct', 'mean'),
        Avg_Quantity=('Quantity', 'mean'), Avg_Days=('Days_In_Pipeline', 'mean'),
        Total_Revenue=('Revenue', 'sum')
    ).round(2)
    st.dataframe(summary, use_container_width=True)

# ═══════════════════════════════════════
# TAB 2: DIAGNOSTIC ANALYSIS
# ═══════════════════════════════════════
with tab2:
    st.header("2️⃣ Diagnostic Analysis — Bias & Inefficiency Detection")

    # Channel analysis
    st.subheader("🔹 Channel-wise Conversion Analysis")
    ch = df_filtered.groupby('Channel').agg(
        Total=('Lead_ID', 'count'), Won=('Deal_Won', 'sum'),
        Win_Rate=('Deal_Won', 'mean'), Avg_Rev=('Revenue', 'mean'),
        Avg_Disc=('Discount_Pct', 'mean')
    ).round(4)
    ch['Win_Rate_Pct'] = (ch['Win_Rate'] * 100).round(2)
    ch = ch.sort_values('Win_Rate_Pct', ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(ch[['Total', 'Won', 'Win_Rate_Pct', 'Avg_Rev', 'Avg_Disc']], use_container_width=True)
        ct_ch = pd.crosstab(df_filtered['Channel'], df_filtered['Deal_Won'])
        chi2, p, _, _ = stats.chi2_contingency(ct_ch)
        if p < 0.05:
            st.markdown(f'<div class="bias-alert">⚠️ Chi-Square: χ²={chi2:.4f}, p={p:.4f} — <b>Statistically significant bias detected</b></div>', unsafe_allow_html=True)
        else:
            st.info(f"Chi-Square: χ²={chi2:.4f}, p={p:.4f} — No statistically significant channel bias")

    with col2:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ch_sorted = ch.sort_values('Total', ascending=True)
        ax1.barh(ch_sorted.index, ch_sorted['Total'], color='#BDD7EE', label='Total Leads')
        ax1.barh(ch_sorted.index, ch_sorted['Won'], color='#2E75B6', label='Deals Won')
        ax2 = ax1.twiny()
        ax2.plot(ch_sorted['Win_Rate_Pct'], ch_sorted.index, 'ro-', lw=2, label='Win Rate %')
        ax1.set_xlabel('Count')
        ax2.set_xlabel('Win Rate %', color='red')
        ax1.set_title('Channel: Volume vs Conversion', fontweight='bold')
        ax1.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Region analysis
    st.subheader("🔹 Region-wise Analysis")
    rg = df_filtered.groupby('Region').agg(
        Total=('Lead_ID', 'count'), Won=('Deal_Won', 'sum'),
        Win_Rate=('Deal_Won', 'mean'), Total_Rev=('Revenue', 'sum'),
        Avg_Rev=('Revenue', 'mean'), Avg_Disc=('Discount_Pct', 'mean')
    ).round(4)
    rg['Win_Rate_Pct'] = (rg['Win_Rate'] * 100).round(2)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(rg.sort_values('Win_Rate_Pct', ascending=False), use_container_width=True)
        ct_rg = pd.crosstab(df_filtered['Region'], df_filtered['Deal_Won'])
        chi2, p, _, _ = stats.chi2_contingency(ct_rg)
        if p < 0.05:
            st.markdown(f'<div class="bias-alert">⚠️ Chi-Square: χ²={chi2:.4f}, p={p:.4f} — <b>Significant regional bias</b></div>', unsafe_allow_html=True)
        else:
            st.info(f"Chi-Square: χ²={chi2:.4f}, p={p:.4f} — No significant regional bias")

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        rg_sorted = rg.sort_values('Win_Rate_Pct', ascending=False)
        ax.bar(rg_sorted.index, rg_sorted['Win_Rate_Pct'],
               color=['#2E75B6', '#4A90D9', '#6BAED6', '#9ECAE1', '#BDD7EE'])
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Region-wise Win Rate', fontweight='bold')
        for i, v in enumerate(rg_sorted['Win_Rate_Pct']):
            ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Sales Rep analysis
    st.subheader("🔹 Sales Rep Performance Analysis")
    rep = df_filtered.groupby('Sales_Rep').agg(
        Total=('Lead_ID', 'count'), Won=('Deal_Won', 'sum'),
        Win_Rate=('Deal_Won', 'mean'), Avg_Rev=('Revenue', 'mean'),
        Avg_Disc=('Discount_Pct', 'mean'), Avg_Days=('Days_In_Pipeline', 'mean')
    ).round(4)
    rep['Win_Rate_Pct'] = (rep['Win_Rate'] * 100).round(2)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(rep.sort_values('Win_Rate_Pct', ascending=False), use_container_width=True)
        ct_rp = pd.crosstab(df_filtered['Sales_Rep'], df_filtered['Deal_Won'])
        chi2, p, _, _ = stats.chi2_contingency(ct_rp)
        st.info(f"Chi-Square: χ²={chi2:.4f}, p={p:.4f}")
        groups = [g['Discount_Pct'].values for _, g in df_filtered.groupby('Sales_Rep')]
        if len(groups) > 1:
            f_stat, p_anova = stats.f_oneway(*groups)
            st.info(f"ANOVA (Discount across Reps): F={f_stat:.4f}, p={p_anova:.4f}")

    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        rep_sorted = rep.sort_values('Win_Rate_Pct', ascending=False)
        avg_wr = df_filtered['Deal_Won'].mean() * 100
        clrs = ['#2E75B6' if v > avg_wr else '#FF6B35' for v in rep_sorted['Win_Rate_Pct']]
        ax.bar(rep_sorted.index, rep_sorted['Win_Rate_Pct'], color=clrs)
        ax.axhline(y=avg_wr, color='red', ls='--', label=f'Avg ({avg_wr:.1f}%)')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Sales Rep Win Rate (Orange = Below Avg)', fontweight='bold')
        ax.legend()
        for i, v in enumerate(rep_sorted['Win_Rate_Pct']):
            ax.text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Segment analysis
    st.subheader("🔹 Customer Segment Analysis")
    seg = df_filtered.groupby('Customer_Segment').agg(
        Total=('Lead_ID', 'count'), Won=('Deal_Won', 'sum'),
        Win_Rate=('Deal_Won', 'mean'), Avg_Rev=('Revenue', 'mean'),
        Avg_Disc=('Discount_Pct', 'mean'), Avg_Qty=('Quantity', 'mean')
    ).round(4)
    seg['Win_Rate_Pct'] = (seg['Win_Rate'] * 100).round(2)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(seg, use_container_width=True)
    with col2:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].bar(seg.index, seg['Win_Rate_Pct'], color=['#2E75B6', '#4A90D9', '#6BAED6'])
        axes[0].set_title('Win Rate %', fontweight='bold')
        axes[1].bar(seg.index, seg['Avg_Rev'], color=['#2E75B6', '#4A90D9', '#6BAED6'])
        axes[1].set_title('Avg Revenue', fontweight='bold')
        axes[2].bar(seg.index, seg['Avg_Disc'] * 100, color=['#2E75B6', '#4A90D9', '#6BAED6'])
        axes[2].set_title('Avg Discount %', fontweight='bold')
        for a in axes:
            a.tick_params(axis='x', rotation=25)
        plt.suptitle('Customer Segment Breakdown', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Discount vs Outcome
    st.subheader("🔹 Discount Impact on Deal Outcome")
    won_disc = df_filtered[df_filtered['Deal_Won'] == 1]['Discount_Pct']
    lost_disc = df_filtered[df_filtered['Deal_Won'] == 0]['Discount_Pct']
    if len(won_disc) > 1 and len(lost_disc) > 1:
        t_stat, p_tt = stats.ttest_ind(won_disc, lost_disc)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Discount — Won", f"{won_disc.mean()*100:.2f}%")
            st.metric("Avg Discount — Not Won", f"{lost_disc.mean()*100:.2f}%")
            st.info(f"T-Test: t={t_stat:.4f}, p={p_tt:.4f}")
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(lost_disc, bins=20, alpha=0.6, label='Not Won', color='#FF6B35')
            ax.hist(won_disc, bins=20, alpha=0.6, label='Won', color='#2E75B6')
            ax.set_xlabel('Discount %')
            ax.set_title('Discount Distribution: Won vs Not Won', fontweight='bold')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ═══════════════════════════════════════
# TAB 3: ML MODELS
# ═══════════════════════════════════════
with tab3:
    st.header("3️⃣ Feature Engineering & Classification Models")

    st.subheader("Feature Engineering Steps")
    fe_steps = pd.DataFrame({
        'Step': ['Label Encoding', 'Standard Scaling', 'SMOTE Oversampling',
                 'Derived: Revenue_Per_Unit', 'Derived: Pipeline_Duration_Days',
                 'Derived: High_Value_Deal', 'Dropped Leaky Columns'],
        'Details': [
            'Categorical → numeric (Sales_Rep, Region, Channel, Segment, Category, DayOfWeek)',
            'StandardScaler on all features (mean=0, std=1)',
            'Balanced minority class (Deal_Won=1) from ~6% to 50% in training set',
            'Revenue / Quantity', '(Close_Date − Lead_Date).days',
            '1 if Revenue ≥ 75th percentile', 'Pipeline_Stage, Conversion_Probability, CSAT_Score'
        ]
    })
    st.dataframe(fe_steps, use_container_width=True, hide_index=True)

    st.subheader("Model Training")
    with st.spinner("Training models..."):
        drop_cols = ['Lead_ID', 'Lead_Date', 'Close_Date', 'Pipeline_Stage',
                     'Conversion_Probability', 'CSAT_Score', 'CSAT_Available',
                     'Discount_Band', 'Revenue_Band']
        df_ml = df.drop(columns=[c for c in drop_cols if c in df.columns])
        cat_cols = ['Sales_Rep', 'Region', 'Channel', 'Customer_Segment', 'Product_Category', 'Lead_DayOfWeek']
        for col in cat_cols:
            df_ml[col] = LabelEncoder().fit_transform(df_ml[col])

        X = df_ml.drop('Deal_Won', axis=1)
        y = df_ml['Deal_Won']
        X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        models = {
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced'),
            'Gradient Boosted': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train_sm, y_train_sm)
            y_tr_pred = model.predict(X_train_sm)
            y_te_pred = model.predict(X_test)
            y_te_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_te_proba)

            results[name] = {
                'model': model,
                'train_acc': accuracy_score(y_train_sm, y_tr_pred),
                'test_acc': accuracy_score(y_test, y_te_pred),
                'precision': precision_score(y_test, y_te_pred, zero_division=0),
                'recall': recall_score(y_test, y_te_pred, zero_division=0),
                'f1': f1_score(y_test, y_te_pred, zero_division=0),
                'roc_auc': auc(fpr, tpr), 'fpr': fpr, 'tpr': tpr,
                'cm': confusion_matrix(y_test, y_te_pred)
            }

    comp = pd.DataFrame({
        name: {
            'Train Accuracy': f"{r['train_acc']:.4f}", 'Test Accuracy': f"{r['test_acc']:.4f}",
            'Precision': f"{r['precision']:.4f}", 'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1']:.4f}", 'ROC AUC': f"{r['roc_auc']:.4f}"
        } for name, r in results.items()
    })
    st.dataframe(comp, use_container_width=True)

    st.subheader("Feature Importance (Gradient Boosted)")
    feat_imp = pd.Series(results['Gradient Boosted']['model'].feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    top = feat_imp.head(12)
    ax.barh(top.index[::-1], top.values[::-1], color='#2E75B6')
    ax.set_xlabel('Importance')
    ax.set_title('Top 12 Feature Importances', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════
# TAB 4: MODEL EVALUATION
# ═══════════════════════════════════════
with tab4:
    st.header("4️⃣ Model Evaluation")

    if 'results' not in dir():
        st.warning("Please run models in Tab 3 first.")
    else:
        model_names = list(results.keys())

        st.subheader("Training vs Testing Accuracy")
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(model_names))
        train_accs = [results[n]['train_acc'] for n in model_names]
        test_accs = [results[n]['test_acc'] for n in model_names]
        ax.bar(x - 0.18, train_accs, 0.35, label='Train', color='#2E75B6')
        ax.bar(x + 0.18, test_accs, 0.35, label='Test', color='#FF6B35')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel('Accuracy')
        ax.set_title('Training vs Testing Accuracy', fontweight='bold')
        ax.legend()
        for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
            ax.text(i - 0.18, tr + 0.01, f'{tr:.3f}', ha='center', fontsize=9)
            ax.text(i + 0.18, te + 0.01, f'{te:.3f}', ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Precision, Recall & F1-Score")
        fig, ax = plt.subplots(figsize=(10, 5))
        precs = [results[n]['precision'] for n in model_names]
        recs = [results[n]['recall'] for n in model_names]
        f1s = [results[n]['f1'] for n in model_names]
        w = 0.25
        ax.bar(x - w, precs, w, label='Precision', color='#2E75B6')
        ax.bar(x, recs, w, label='Recall', color='#FF6B35')
        ax.bar(x + w, f1s, w, label='F1-Score', color='#2CA02C')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall & F1-Score', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, name in enumerate(model_names):
            r = results[name]
            ax.plot(r['fpr'], r['tpr'], color=colors4[i], lw=2, label=f"{name} (AUC={r['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves', fontweight='bold')
        ax.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Confusion Matrices")
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for i, name in enumerate(model_names):
            sns.heatmap(results[name]['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['Not Won', 'Won'], yticklabels=['Not Won', 'Won'])
            axes[i].set_title(name, fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        plt.suptitle('Confusion Matrices', fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════
# TAB 5: FINDINGS
# ═══════════════════════════════════════
with tab5:
    st.header("5️⃣ Key Findings & Insights")

    st.subheader("📊 Descriptive Findings")
    st.markdown("""
    - The pipeline is **heavily top-heavy**: 27.1% of all records sit at the Lead stage, with a steep drop-off through subsequent stages.
    - Only **6.3% of leads convert to Closed Won** — a very low overall win rate indicating significant pipeline leakage.
    - Average deal revenue is relatively consistent across stages ($900–$1,260), suggesting that deal value alone is not what determines progression.
    - The average time in the pipeline is about **30 days**, with no major variation by stage.
    """)

    st.subheader("🔍 Diagnostic Findings — Bias Analysis")
    st.markdown("""
    **Channel-wise:** No statistically significant bias was found (Chi-Square p=0.94). However, **Direct channel** shows the highest win rate (8.3%) despite having the lowest lead volume, while **Referral** has the lowest (5.0%). This suggests Direct leads may be warmer/better qualified.

    **Region-wise:** The **East region** leads with a 10.6% win rate — nearly double the North region (4.5%). Although the Chi-Square test showed borderline significance (p=0.078), this warrants closer monitoring for potential regional allocation bias.

    **Sales Rep-wise:** Win rates range from **3.7% (Rep_04) to 8.7% (Rep_08)** — a 2.4x gap. However, this is not statistically significant (p=0.88), and the ANOVA test showed **no significant variation in discounting behaviour** across reps (p=0.99).

    **Customer Segment-wise:** New Customers actually convert slightly better (7.2%) than Returning Customers (4.9%), though VIP Customers generate the highest average revenue ($1,394). No statistically significant bias was found (p=0.36).
    """)

    st.subheader("🤖 Model Findings")
    st.markdown("""
    - All four models achieved **high overall accuracy (93–94%)**, but this is largely driven by correctly predicting the dominant "Not Won" class.
    - **Precision, Recall, and F1-Scores were very low** across all models, indicating that the available features are **weak predictors of deal closure**.
    - ROC AUC scores hover around **0.43–0.58** (near random), confirming that the current feature set doesn't capture the real drivers of deal conversion.
    - The **top features by importance** were Revenue, Days in Pipeline, Revenue Per Unit, and Discount — all transactional features rather than behavioural ones.
    """)

    st.subheader("💡 Recommendations")
    st.markdown("""
    1. **Enrich the dataset** with behavioural features: number of follow-up calls, email response times, demo attendance, website engagement — these are likely the true conversion drivers.
    2. **Investigate the East region's outperformance** — is it due to better reps, stronger demand, or different product mix?
    3. **Address pipeline leakage** — 73% of leads never progress past Prospect stage. Implement lead scoring to focus rep effort on high-potential leads.
    4. **Standardise the Direct channel playbook** — its higher conversion rate could be replicated if the qualification criteria are documented and applied to other channels.
    5. **Monitor Rep_04 and Rep_01** — their below-average win rates may indicate training needs or suboptimal lead assignment.
    """)

    st.info("Note: The low model performance is itself a key finding — it confirms that deal closure in this startup's pipeline is NOT systematically biased by channel, region, segment, or rep. The randomness suggests external factors (customer intent, product-market fit, follow-up quality) are the real drivers.")
