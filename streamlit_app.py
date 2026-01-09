import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Insurance Claims Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean look
st.markdown("""
<style>
    /* Light modern theme */
    .main {
        background-color: #ffffff; /* pure white for a clean look */
        color: #212529; /* dark text for readability */
    }
    .stMetric {
        background-color: #f8f9fa; /* subtle gray cards */
        padding: 18px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        background-color: #ffffff;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Simple typography */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def generate_data_if_missing():
    """Generate data files if they don't exist."""
    data_files = ['data/claims_master.csv', 'data/customers.csv', 'data/fraud_scores.csv']
    
    if all(os.path.exists(f) for f in data_files):
        return True
    
    # Create data directory if needed
    os.makedirs('data', exist_ok=True)
    
    try:
        from data_generator import InsuranceDataGenerator
        from fraud_detection import FraudDetectionModel
        
        with st.spinner('Generating data for first-time setup... This may take a minute.'):
            # Generate synthetic data
            generator = InsuranceDataGenerator(n_customers=5000, n_claims=12000)
            customers_df, claims_df = generator.export_to_csv('data')
            
            # Train model and generate scores
            model = FraudDetectionModel()
            model.train(claims_df)
            model.export_predictions(claims_df, 'data/fraud_scores.csv')
            model.export_feature_importance('data/feature_importance.csv')
        
        return True
    except Exception as e:
        st.error(f"Failed to generate data: {e}")
        return False

# Data loading with caching
@st.cache_data
def load_data():
    try:
        claims = pd.read_csv('data/claims_master.csv')
        customers = pd.read_csv('data/customers.csv')
        fraud_scores = pd.read_csv('data/fraud_scores.csv')
        
        # Merge claims and fraud scores
        df = pd.merge(claims, fraud_scores, on='claim_id')
        df['claim_date'] = pd.to_datetime(df['claim_date'])
        
        return df, customers
    except FileNotFoundError:
        return None, None

# Generate data if missing, then load
if not generate_data_if_missing():
    st.stop()

df, customers = load_data()

if df is None:
    st.error("Failed to load data. Please check the logs.")
    st.stop()

if df is not None:
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Executive Overview", "Fraud Detection Analysis", "Customer Insights", "Data Explorer"])
    
    st.sidebar.divider()
    
    with st.sidebar.expander("🔍 Filters & Settings", expanded=True):
        date_range = st.date_input(
            "Date Range",
            value=(df['claim_date'].min(), df['claim_date'].max()),
            min_value=df['claim_date'].min(),
            max_value=df['claim_date'].max()
        )
        
        policy_types = st.multiselect("Policy Type", options=df['policy_type'].unique(), default=df['policy_type'].unique())
    
    # Filter data based on sidebar selections
    filtered_df = df.copy()
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (filtered_df['claim_date'].dt.date >= start_date) & (filtered_df['claim_date'].dt.date <= end_date)
        filtered_df = filtered_df.loc[mask]

    if policy_types:
        filtered_df = filtered_df[filtered_df['policy_type'].isin(policy_types)]
        
    if filtered_df.empty:
        st.warning("No data found for the selected filters. Please adjust your selection.")
        st.stop()

    # --- Page 1: Executive Overview ---
    # --- Page 1: Executive Overview ---
    if page == "Executive Overview":
        st.title("Executive Overview")
        
        # Calculate key metrics
        total_claims = len(filtered_df)
        total_payout = filtered_df['payout_amount'].sum()
        avg_payout = filtered_df['payout_amount'].mean() if total_claims > 0 else 0
        avg_processing = filtered_df['processing_days'].mean() if total_claims > 0 else 0
        approval_rate = (filtered_df['claim_status'] == 'Approved').mean() * 100 if total_claims > 0 else 0
        sla_breach_rate = (filtered_df['processing_days'] > 14).mean() * 100 if total_claims > 0 else 0
        
        # Calculate deltas (vs previous period of same length)
        # Simplified: compare to previous window of same duration
        if len(date_range) == 2:
            duration = (end_date - start_date).days
            prev_start = start_date - pd.Timedelta(days=duration)
            prev_end = start_date
            prev_mask = (df['claim_date'].dt.date >= prev_start) & (df['claim_date'].dt.date < prev_end)
            prev_df = df.loc[prev_mask]
            
            if len(prev_df) > 0:
                prev_claims = len(prev_df)
                prev_payout = prev_df['payout_amount'].sum()
                claims_delta = ((total_claims - prev_claims) / prev_claims) * 100
                payout_delta = ((total_payout - prev_payout) / prev_payout) * 100
            else:
                claims_delta, payout_delta = 0, 0
        else:
            claims_delta, payout_delta = 0, 0

        # KPI Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Claims", f"{total_claims:,}", f"{claims_delta:.1f}%")
        c2.metric("Total Payout", f"${total_payout/1e6:.2f}M", f"{payout_delta:.1f}%")
        c3.metric("Avg Payout", f"${avg_payout:.0f}")
        c4.metric("SLA Breach Rate (>14 days)", f"{sla_breach_rate:.1f}%", delta_color="inverse")
        
        st.divider()

        # Charts: Volume & Payout Trend (Dual Axis)
        st.subheader("Claims Volume & Payout Trends")
        
        # Aggregate by month
        monthly_agg = filtered_df.groupby(filtered_df['claim_date'].dt.to_period('M')).agg({
            'claim_id': 'count',
            'payout_amount': 'sum'
        }).reset_index()
        monthly_agg['period'] = monthly_agg['claim_date'].astype(str)
        
        # Create dual-axis chart
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly_agg['period'], 
            y=monthly_agg['claim_id'], 
            name='Claims Volume',
            marker_color='#3b82f6'
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_agg['period'], 
            y=monthly_agg['payout_amount'], 
            name='Total Payout ($)',
            yaxis='y2',
            line=dict(color='#ef4444', width=3)
        ))
        
        fig_trend.update_layout(
            yaxis=dict(title="Claims Volume"),
            yaxis2=dict(title="Total Payout ($)", overlaying='y', side='right'),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Row 2: Status Funnel & Geographic Map
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader("Claims Status Funnel")
            status_counts = filtered_df['claim_status'].value_counts().reset_index()
            # Order intuitively
            order = ['Received', 'Under Review', 'Approved', 'Denied', 'Closed'] # Simplified logic
            fig_funnel = px.funnel(status_counts, y='claim_status', x='count', 
                                  title="Claims Processing Pipeline")
            st.plotly_chart(fig_funnel, use_container_width=True)
            
        with c_right:
            st.subheader("Geographic Heatmap (Avg Payout)")
            prov_stats = filtered_df.groupby('province')['payout_amount'].mean().reset_index()
            fig_map = px.choropleth(
                prov_stats, 
                locations='province', 
                locationmode="USA-states", # Fallback, usually need geojson for Canada or proper scope
                scope="north america",
                color='payout_amount',
                title="Avg Payout by Province"
            )
            # Since standard plotly maps for Canada can be tricky without geojson, let's use a nice bar chart if map fails expectations, 
            # but let's try a colored bar chart that looks like a distribution
            fig_geo = px.bar(prov_stats.sort_values('payout_amount', ascending=False), 
                            x='province', y='payout_amount', color='payout_amount',
                            title="Avg Payout Intensity by Province")
            st.plotly_chart(fig_geo, use_container_width=True)

    # --- Page 2: Fraud Detection Analysis ---
    # --- Page 2: Fraud Detection Analysis ---
    elif page == "Fraud Detection Analysis":
        st.title("ML Fraud Detection Analysis")
        st.markdown("Insights from the Isolation Forest + Random Forest ensemble model.")
        
        # 1. Trend Analysis (New)
        st.subheader("Suspicious Activity Trend")
        daily_fraud = filtered_df[filtered_df['risk_category'].isin(['High', 'Critical'])].groupby('claim_date').size().reset_index(name='count')
        if not daily_fraud.empty:
            fig_trend = px.area(daily_fraud, x='claim_date', y='count', 
                               title="Volume of High-Risk Claims Over Time",
                               labels={'count': 'Number of High Risk Claims'},
                               color_discrete_sequence=['#ef4444'])
            fig_trend.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No high-risk claims found in the selected period.")

        # 2. Distribution & Correlation
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Risk Distribution")
            risk_dist = filtered_df['risk_category'].value_counts().reset_index()
            fig_risk = px.pie(risk_dist, values='count', names='risk_category', 
                               hole=0.4,
                               color='risk_category',
                               color_discrete_map={'Critical': '#dc2626', 'High': '#ea580c', 'Medium': '#0891b2', 'Low': '#16a34a'})
            st.plotly_chart(fig_risk, use_container_width=True)
            
        with c2:
            st.subheader("Risk Score vs Claim Amount")
            fig_scatter = px.scatter(filtered_df, x='claim_amount', y='risk_score', 
                                    color='risk_category', size='amount_vs_avg_ratio',
                                    hover_data=['claim_id', 'claim_type', 'province'],
                                    title="Sensitivity Analysis: Value vs. Detected Risk")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.divider()
        
        # 3. Actionable Insights
        st.subheader("Top Suspicious Claims for Review")
        
        # Filter for high risk
        suspicious_df = filtered_df[filtered_df['risk_score'] > 75].sort_values('risk_score', ascending=False).head(20)
        
        if not suspicious_df.empty:
            # Create a more readable view
            display_cols = ['claim_id', 'claim_date', 'claim_type', 'claim_amount', 'risk_score', 'risk_category', 'province']
            
            # Use styling for the dataframe
            st.dataframe(
                suspicious_df[display_cols].style.background_gradient(subset=['risk_score'], cmap='Reds'),
                use_container_width=True
            )
            
            # Model explainability in an expander to save space
            with st.expander("Explanation: Top Fraud Indicators"):
                if os.path.exists('data/feature_importance.csv'):
                    importance_df = pd.read_csv('data/feature_importance.csv').head(10)
                    fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h', 
                                     title="Model Feature Importance",
                                     labels={'importance': 'Impact Weight', 'feature': 'Indicator'})
                    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.success("No claims require immediate review based on current sensitivity settings.")

    # --- Page 3: Customer Insights ---
    elif page == "Customer Insights":
        st.title("Customer Insights")
        
        # Merge data for analysis
        cust_claims = pd.merge(filtered_df, customers, on='customer_id')
        
        # Advanced Metrics
        if not cust_claims.empty:
            avg_tenure = (pd.to_datetime('today') - pd.to_datetime(cust_claims['customer_since'])).dt.days.mean()
            avg_claims_per_cust = len(cust_claims) / cust_claims['customer_id'].nunique()
            retention_rate = (cust_claims['claim_date'].max() > pd.Timestamp.now() - pd.DateOffset(months=12)) # simple proxy
            top_tier = cust_claims['premium_tier'].mode()[0]
        else:
            avg_tenure, avg_claims_per_cust, top_tier = 0, 0, "N/A"

        # Key Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Tenure", f"{int(avg_tenure/365)} Years")
        m2.metric("Claims / Customer", f"{avg_claims_per_cust:.2f}")
        m3.metric("Most Common Tier", top_tier)
        m4.metric("Active Customers", f"{cust_claims['customer_id'].nunique():,}")
        
        st.divider()
        
        # Row 1: Demographics & Tier
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Demographics Analysis")
            # Binning ages dynamically
            if 'age' in cust_claims.columns:
                cust_claims['age_group'] = pd.cut(cust_claims['age'], bins=[18, 30, 45, 60, 100], labels=['18-30', '30-45', '45-60', '60+'])
                fig_age = px.histogram(cust_claims, x='age_group', color='claim_type', 
                                      title="Claim Volume by Age Group & Type",
                                      barmode='stack')
                st.plotly_chart(fig_age, use_container_width=True)
            
        with c2:
            st.subheader("Premium Tier Performance")
            tier_stats = cust_claims.groupby('premium_tier').agg({
                'claim_amount': 'mean',
                'customer_id': 'nunique'
            }).reset_index()
            fig_tier = px.bar(tier_stats, x='premium_tier', y='claim_amount', 
                             title="Avg Claim Value by Tier",
                             labels={'claim_amount': 'Avg Claim ($)'},
                             color='premium_tier')
            st.plotly_chart(fig_tier, use_container_width=True)
        
        # Row 2: Behavioral Analysis
        st.subheader("Behavioral Segmentation")
        b1, b2 = st.columns(2)
        
        with b1:
            # Tenure vs Claims (Loyalty)
            cust_aggregated = cust_claims.groupby('customer_id').agg({
                'customer_since': 'min', # just to get one value
                'claim_id': 'count',
                'claim_amount': 'sum',
                'premium_tier': 'first',
                'credit_score': 'first'
            }).reset_index()
            
            cust_aggregated['tenure_days'] = (pd.to_datetime('today') - pd.to_datetime(cust_aggregated['customer_since'])).dt.days
            
            fig_loyalty = px.scatter(cust_aggregated, x='tenure_days', y='claim_amount', 
                                    color='premium_tier', size='claim_id',
                                    title="Customer Value vs Tenure (Bubble Size = Claim Count)",
                                    labels={'tenure_days': 'Tenure (Days)', 'claim_amount': 'Total Lifetime Value ($)'},
                                    hover_data=['customer_id'])
            st.plotly_chart(fig_loyalty, use_container_width=True)
            
        with b2:
            # Credit Score correlation
            fig_credit = px.box(cust_claims, x='premium_tier', y='credit_score', 
                               points="all",
                               title="Credit Score Distribution by Tier",
                               color='premium_tier')
            st.plotly_chart(fig_credit, use_container_width=True)

        # Row 3: Top Customers Table
        with st.expander("🏆 Top High-Value Customers", expanded=True):
            top_cust = cust_aggregated.sort_values('claim_amount', ascending=False).head(15)
            st.dataframe(
                top_cust[['customer_id', 'premium_tier', 'credit_score', 'claim_id', 'claim_amount', 'tenure_days']].style.format({
                    'claim_amount': '${:,.2f}',
                    'tenure_days': '{:,.0f}'
                }),
                use_container_width=True
            )

    # --- Page 4: Data Explorer ---
    elif page == "Data Explorer":
        st.title("Raw Data Explorer")
        st.markdown("Search, filter, and export detailed claims data.")
        
        # Local Filters
        with st.expander("📂 Advanced Filters", expanded=True):
            f1, f2, f3 = st.columns(3)
            with f1:
                status_filter = st.multiselect("Claim Status", options=filtered_df['claim_status'].unique())
            with f2:
                risk_filter = st.multiselect("Risk Category", options=filtered_df['risk_category'].unique())
            with f3:
                min_amt, max_amt = filtered_df['claim_amount'].min(), filtered_df['claim_amount'].max()
                amt_range = st.slider("Claim Amount Range", float(min_amt), float(max_amt), (float(min_amt), float(max_amt)))
        
        # Apply filters
        explorer_df = filtered_df.copy()
        if status_filter:
            explorer_df = explorer_df[explorer_df['claim_status'].isin(status_filter)]
        if risk_filter:
            explorer_df = explorer_df[explorer_df['risk_category'].isin(risk_filter)]
            
        explorer_df = explorer_df[
            (explorer_df['claim_amount'] >= amt_range[0]) & 
            (explorer_df['claim_amount'] <= amt_range[1])
        ]
        
        # Search functionality
        search_id = st.text_input("Search by Claim ID (e.g. CLM0000001)")
        if search_id:
            explorer_df = explorer_df[explorer_df['claim_id'].str.contains(search_id, case=False)]
            
        # Display Data
        st.subheader(f"Filtered Results: {len(explorer_df):,} Claims")
        
        # Download Button
        csv = explorer_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Filtered Data (CSV)",
            data=csv,
            file_name=f"claims_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
        
        # Main Table
        st.dataframe(explorer_df, use_container_width=True, height=500)
        
        # Quick Stats for selection
        if not explorer_df.empty:
            st.divider()
            st.caption("Quick Stats for Selection:")
            qs1, qs2, qs3 = st.columns(3)
            qs1.metric("Total Value", f"${explorer_df['claim_amount'].sum()/1e6:.2f}M")
            qs2.metric("Avg Risk Score", f"{explorer_df['risk_score'].mean():.1f}")
            qs3.metric("Critical Risks", len(explorer_df[explorer_df['risk_category']=='Critical']))

# Footer info
st.sidebar.divider()
st.sidebar.info("Developed for TD Insurance Portfolio Project")
