"""
Fraud Detection Model

Machine Learning model using Isolation Forest and ensemble techniques
to detect fraudulent insurance claims.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionModel:
    """
    ML-based fraud detection using Isolation Forest (unsupervised) 
    and Random Forest (supervised) ensemble approach.
    """
    
    # Features for fraud detection
    FEATURE_COLUMNS = [
        'claim_amount',
        'claim_frequency_90d',
        'amount_vs_avg_ratio',
        'weekend_filing',
        'late_reporting_days',
        'documentation_score',
        'witness_present',
        'police_report_filed',
        'previous_claims_count',
        'policy_age_days',
        'processing_days'
    ]
    
    def __init__(self):
        self.isolation_forest = None
        self.random_forest = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        
    def prepare_features(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features for ML model."""
        df = claims_df.copy()
        
        # Encode categorical variables
        categorical_cols = ['policy_type', 'claim_type', 'claim_status', 'province']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Additional feature engineering
        df['amount_per_processing_day'] = df['claim_amount'] / (df['processing_days'] + 1)
        df['high_amount_flag'] = (df['claim_amount'] > df['claim_amount'].quantile(0.9)).astype(int)
        df['new_customer_flag'] = (df['policy_age_days'] < 180).astype(int)
        df['low_doc_score_flag'] = (df['documentation_score'] < 70).astype(int)
        
        # Risk composite score
        df['risk_composite'] = (
            df['amount_vs_avg_ratio'] * 0.3 +
            df['claim_frequency_90d'] * 0.2 +
            (100 - df['documentation_score']) / 100 * 0.2 +
            df['late_reporting_days'] / 20 * 0.15 +
            (1 - df['police_report_filed']) * 0.15
        )
        
        return df
    
    def train(self, claims_df: pd.DataFrame, use_labels: bool = True):
        """
        Train the fraud detection model.
        
        Args:
            claims_df: DataFrame with claims data
            use_labels: If True and 'is_fraud' exists, use supervised learning
        """
        df = self.prepare_features(claims_df)
        
        # Select features
        feature_cols = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        feature_cols.extend(['risk_composite', 'amount_per_processing_day', 
                           'high_amount_flag', 'new_customer_flag', 'low_doc_score_flag'])
        feature_cols.extend([f'{col}_encoded' for col in ['policy_type', 'claim_type', 'province'] 
                           if f'{col}_encoded' in df.columns])
        
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest (unsupervised anomaly detection)
        print("Training Isolation Forest model...")
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=0.07,  # Expected ~7% fraud rate
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)
        
        # If labels available, also train supervised model
        if use_labels and 'is_fraud' in claims_df.columns:
            print("Training Random Forest classifier...")
            y = claims_df['is_fraud'].astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.random_forest = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.random_forest.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.random_forest.predict(X_test)
            y_prob = self.random_forest.predict_proba(X_test)[:, 1]
            
            print("\n=== Model Evaluation ===")
            print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
            
            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.random_forest.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10).to_string(index=False))
        
        self.feature_columns = feature_cols
        print("\n✓ Model training complete!")
        
    def predict_risk_scores(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate fraud risk scores for claims.
        
        Returns DataFrame with claim_id and risk_score (0-100).
        """
        df = self.prepare_features(claims_df)
        
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        results = pd.DataFrame({'claim_id': claims_df['claim_id']})
        
        # Isolation Forest anomaly score (-1 to 1, convert to 0-100)
        if self.isolation_forest:
            iso_scores = self.isolation_forest.decision_function(X_scaled)
            # More negative = more anomalous, convert to 0-100 risk score
            results['anomaly_score'] = (1 - (iso_scores - iso_scores.min()) / 
                                        (iso_scores.max() - iso_scores.min())) * 100
        
        # Random Forest probability if available
        if self.random_forest:
            rf_proba = self.random_forest.predict_proba(X_scaled)[:, 1]
            results['ml_fraud_probability'] = rf_proba * 100
            
            # Ensemble score (weighted average)
            results['risk_score'] = (
                results['anomaly_score'] * 0.4 + 
                results['ml_fraud_probability'] * 0.6
            ).round(1)
        else:
            results['risk_score'] = results['anomaly_score'].round(1)
        
        # Risk category
        results['risk_category'] = pd.cut(
            results['risk_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return results
    
    def export_predictions(self, claims_df: pd.DataFrame, output_path: str = 'data/fraud_scores.csv'):
        """Export fraud risk scores to CSV for Power BI."""
        risk_scores = self.predict_risk_scores(claims_df)
        
        # Add additional context for Power BI
        risk_scores['flagged_for_review'] = risk_scores['risk_score'] >= 70
        risk_scores['requires_investigation'] = risk_scores['risk_score'] >= 85
        
        risk_scores.to_csv(output_path, index=False)
        print(f"✓ Exported fraud scores to {output_path}")
        
        # Summary stats
        print(f"\n=== Risk Score Summary ===")
        print(risk_scores['risk_category'].value_counts().sort_index())
        print(f"\nClaims flagged for review: {risk_scores['flagged_for_review'].sum()}")
        print(f"Claims requiring investigation: {risk_scores['requires_investigation'].sum()}")
        
        return risk_scores
    
    def export_feature_importance(self, output_path: str = 'data/feature_importance.csv'):
        """Export feature importance for Power BI visualization."""
        if self.feature_importance is not None:
            self.feature_importance.to_csv(output_path, index=False)
            print(f"✓ Exported feature importance to {output_path}")


if __name__ == '__main__':
    # Example usage
    print("Loading claims data...")
    claims = pd.read_csv('data/claims_master.csv')
    
    # For demo, add back the is_fraud column (in real scenario, this is for training only)
    from data_generator import InsuranceDataGenerator
    gen = InsuranceDataGenerator()
    _, full_claims = gen.export_to_csv()
    
    model = FraudDetectionModel()
    model.train(full_claims)
    model.export_predictions(full_claims)
    model.export_feature_importance()
