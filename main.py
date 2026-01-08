"""
Insurance Claims Analytics & Fraud Detection
Main Entry Point

Runs the complete data pipeline:
1. Generate synthetic claims data
2. Train ML fraud detection model
3. Export CSV datasets for Dashboard
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import InsuranceDataGenerator
from fraud_detection import FraudDetectionModel


def main():
    print("=" * 60)
    print("  Insurance Claims Analytics & Fraud Detection Pipeline")
    print("  TD Insurance Data Analytics Project")
    print("=" * 60)
    
    # Step 1: Generate Data
    print("\n📊 Step 1: Generating synthetic insurance data...")
    generator = InsuranceDataGenerator(n_customers=5000, n_claims=12000)
    customers_df, claims_df = generator.export_to_csv('data')
    
    print(f"\n📈 Data Summary:")
    print(f"   • Customers: {len(customers_df):,}")
    print(f"   • Claims: {len(claims_df):,}")
    print(f"   • Fraud rate: {claims_df['is_fraud'].mean()*100:.1f}%")
    
    # Step 2: Train ML Model
    print("\n🤖 Step 2: Training fraud detection ML model...")
    model = FraudDetectionModel()
    model.train(claims_df)
    
    # Step 3: Generate Predictions
    print("\n🔍 Step 3: Generating fraud risk scores...")
    risk_scores = model.export_predictions(claims_df, 'data/fraud_scores.csv')
    model.export_feature_importance('data/feature_importance.csv')
    
    # Summary
    print("\n🚀 DONE! Project is ready for your TD Insurance application.")
    print("-" * 50)
    print("Next Steps:")
    print("1. LAUNCH DASHBOARD: Run 'streamlit run streamlit_app.py'")
    print("2. REVIEW SQL: Check 'sql/' directory for analytics scripts")
    print("3. REVIEW DATA: Check 'data/' for processed CSVs")
    print("-" * 50)
    print("\n📁 Output files in data/ folder:")
    print("   • customers.csv - Customer demographics")
    print("   • claims_master.csv - Insurance claims")
    print("   • fraud_scores.csv - ML risk predictions")
    print("   • feature_importance.csv - Top fraud indicators")
    print("   • date_dimension.csv - Date dimension table")
    # Updated next steps focusing on Streamlit
    print("\n📊 Next Steps:")
    print("   1. Launch Streamlit Dashboard: run 'streamlit run streamlit_app.py'")
    print("   2. Review generated CSVs in the 'data/' folder for further analysis.")
    print("   3. Explore SQL scripts in the 'sql/' directory for deeper insights.")
    
    return customers_df, claims_df, risk_scores


if __name__ == '__main__':
    main()
