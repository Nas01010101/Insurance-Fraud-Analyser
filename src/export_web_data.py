"""
Export sample data as JSON for web dashboard
"""

import pandas as pd
import json
import os


def export_dashboard_data():
    """Export aggregated data as JSON for web dashboard."""
    
    os.makedirs('docs/data', exist_ok=True)
    
    # Load data
    claims = pd.read_csv('data/claims_master.csv')
    customers = pd.read_csv('data/customers.csv')
    fraud_scores = pd.read_csv('data/fraud_scores.csv')
    
    # Merge for analysis
    df = claims.merge(fraud_scores, on='claim_id')
    
    # 1. Overview stats
    overview = {
        'totalClaims': len(claims),
        'totalCustomers': len(customers),
        'totalClaimAmount': round(claims['claim_amount'].sum(), 2),
        'totalPayout': round(claims['payout_amount'].sum(), 2),
        'avgClaimAmount': round(claims['claim_amount'].mean(), 2),
        'avgProcessingDays': round(claims['processing_days'].mean(), 1),
        'approvalRate': round((claims['claim_status'] == 'Approved').mean() * 100, 1),
        'highRiskCount': int((fraud_scores['risk_category'].isin(['High', 'Critical'])).sum()),
        'flaggedAmount': round(df[df['flagged_for_review'] == True]['claim_amount'].sum(), 2)
    }
    
    # 2. Claims by type
    claims_by_type = claims.groupby('claim_type').agg({
        'claim_id': 'count',
        'claim_amount': 'sum'
    }).reset_index()
    claims_by_type.columns = ['type', 'count', 'amount']
    claims_by_type = claims_by_type.to_dict('records')
    
    # 3. Monthly trend
    claims['month'] = pd.to_datetime(claims['claim_date']).dt.to_period('M').astype(str)
    monthly = claims.groupby('month').agg({
        'claim_id': 'count',
        'claim_amount': 'sum'
    }).reset_index()
    monthly.columns = ['month', 'count', 'amount']
    monthly = monthly.sort_values('month').tail(12).to_dict('records')
    
    # 4. Risk distribution
    risk_dist = fraud_scores['risk_category'].value_counts().to_dict()
    
    # 5. Claims by province
    province_data = claims.groupby('province').agg({
        'claim_id': 'count',
        'claim_amount': 'sum'
    }).reset_index()
    province_data.columns = ['province', 'count', 'amount']
    province_data = province_data.to_dict('records')
    
    # 6. Top high-risk claims (sample for dashboard)
    high_risk = df[df['risk_score'] >= 70][['claim_id', 'claim_type', 'claim_amount', 
                                             'risk_score', 'risk_category']].head(20)
    high_risk = high_risk.to_dict('records')
    
    # 7. Feature importance (from model)
    try:
        importance = pd.read_csv('data/feature_importance.csv')
        feature_importance = importance.head(10).to_dict('records')
    except:
        feature_importance = []
    
    # 8. Policy type split
    policy_split = claims['policy_type'].value_counts().to_dict()
    
    # 9. Status distribution
    status_dist = claims['claim_status'].value_counts().to_dict()
    
    # Combine all
    dashboard_data = {
        'overview': overview,
        'claimsByType': claims_by_type,
        'monthlyTrend': monthly,
        'riskDistribution': risk_dist,
        'provinceData': province_data,
        'highRiskClaims': high_risk,
        'featureImportance': feature_importance,
        'policySplit': policy_split,
        'statusDistribution': status_dist
    }
    
    # Export
    with open('docs/data/dashboard.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print("✓ Exported dashboard data to docs/data/dashboard.json")
    return dashboard_data


if __name__ == '__main__':
    export_dashboard_data()
