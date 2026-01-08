"""
Insurance Claims Data Generator

Generates realistic synthetic data for auto and home insurance claims
with embedded fraud patterns for ML detection.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker('en_CA')  # Canadian locale for TD Insurance context
Faker.seed(42)
np.random.seed(42)
random.seed(42)


class InsuranceDataGenerator:
    """Generates synthetic insurance claims data with fraud patterns."""
    
    # Canadian provinces with realistic distribution
    PROVINCES = {
        'ON': 0.38, 'QC': 0.23, 'BC': 0.13, 'AB': 0.12,
        'MB': 0.04, 'SK': 0.03, 'NS': 0.03, 'NB': 0.02,
        'NL': 0.01, 'PE': 0.01
    }
    
    # Claim types
    AUTO_CLAIM_TYPES = ['Collision', 'Comprehensive', 'Liability', 'Theft', 'Vandalism']
    HOME_CLAIM_TYPES = ['Water Damage', 'Fire', 'Theft', 'Weather', 'Liability', 'Vandalism']
    
    def __init__(self, n_customers: int = 5000, n_claims: int = 12000):
        self.n_customers = n_customers
        self.n_claims = n_claims
        self.customers_df = None
        self.claims_df = None
        
    def generate_customers(self) -> pd.DataFrame:
        """Generate customer demographics."""
        customers = []
        
        provinces = list(self.PROVINCES.keys())
        province_weights = list(self.PROVINCES.values())
        
        for i in range(self.n_customers):
            age = int(np.random.choice(
                range(18, 85),
                p=self._age_distribution()
            ))
            province = np.random.choice(provinces, p=province_weights)
            
            customer = {
                'customer_id': f'CUS{i+1:06d}',
                'first_name': fake.first_name(),
                'last_name': fake.last_name(),
                'email': fake.email(),
                'phone': fake.phone_number(),
                'date_of_birth': fake.date_of_birth(minimum_age=age, maximum_age=age),
                'age': age,
                'gender': np.random.choice(['M', 'F', 'Other'], p=[0.48, 0.48, 0.04]),
                'province': province,
                'city': fake.city(),
                'postal_code': fake.postalcode(),
                'customer_since': fake.date_between(start_date='-15y', end_date='-30d'),
                'credit_score': int(np.clip(np.random.normal(700, 80), 300, 850)),
                'has_auto_policy': np.random.random() < 0.75,
                'has_home_policy': np.random.random() < 0.45,
                'premium_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 
                                                  p=[0.35, 0.40, 0.20, 0.05])
            }
            customers.append(customer)
        
        self.customers_df = pd.DataFrame(customers)
        return self.customers_df
    
    def generate_claims(self) -> pd.DataFrame:
        """Generate insurance claims with embedded fraud patterns."""
        if self.customers_df is None:
            self.generate_customers()
        
        claims = []
        claim_date_start = datetime.now() - timedelta(days=730)  # 2 years of data
        
        # Customers with policies
        auto_customers = self.customers_df[self.customers_df['has_auto_policy']]['customer_id'].tolist()
        home_customers = self.customers_df[self.customers_df['has_home_policy']]['customer_id'].tolist()
        
        for i in range(self.n_claims):
            # Decide if auto or home claim
            is_auto = np.random.random() < 0.65  # 65% auto claims
            
            if is_auto and auto_customers:
                customer_id = np.random.choice(auto_customers)
                claim_type = np.random.choice(self.AUTO_CLAIM_TYPES, 
                                              p=[0.40, 0.25, 0.20, 0.10, 0.05])
                policy_type = 'Auto'
                base_amount = self._generate_auto_claim_amount(claim_type)
            elif home_customers:
                customer_id = np.random.choice(home_customers)
                claim_type = np.random.choice(self.HOME_CLAIM_TYPES,
                                              p=[0.35, 0.15, 0.20, 0.15, 0.10, 0.05])
                policy_type = 'Home'
                base_amount = self._generate_home_claim_amount(claim_type)
            else:
                continue
            
            customer = self.customers_df[self.customers_df['customer_id'] == customer_id].iloc[0]
            
            # Generate claim date with seasonality
            claim_date = self._generate_claim_date(claim_date_start, claim_type)
            
            # Determine if this is a fraudulent claim (5-8% fraud rate)
            is_fraud = np.random.random() < 0.065
            
            # Fraud patterns affect claim characteristics
            if is_fraud:
                claim_amount, fraud_indicators = self._apply_fraud_patterns(
                    base_amount, claim_type, customer
                )
            else:
                claim_amount = base_amount * np.random.uniform(0.8, 1.2)
                fraud_indicators = self._generate_normal_indicators()
            
            # Processing details
            processing_days = int(np.random.exponential(15) + 3)
            if is_fraud:
                processing_days += int(np.random.exponential(10))  # Fraud takes longer
            
            claim = {
                'claim_id': f'CLM{i+1:07d}',
                'customer_id': customer_id,
                'policy_type': policy_type,
                'claim_type': claim_type,
                'claim_date': claim_date.strftime('%Y-%m-%d'),
                'claim_amount': round(claim_amount, 2),
                'deductible': self._get_deductible(policy_type, customer['premium_tier']),
                'payout_amount': round(max(0, claim_amount - self._get_deductible(policy_type, customer['premium_tier'])), 2),
                'claim_status': np.random.choice(
                    ['Approved', 'Denied', 'Under Review', 'Closed'],
                    p=[0.65, 0.15, 0.10, 0.10]
                ),
                'processing_days': processing_days,
                'adjuster_id': f'ADJ{np.random.randint(1, 51):03d}',
                'province': customer['province'],
                
                # Fraud indicators (features for ML)
                'claim_frequency_90d': fraud_indicators['claim_frequency'],
                'amount_vs_avg_ratio': fraud_indicators['amount_ratio'],
                'weekend_filing': fraud_indicators['weekend_filing'],
                'late_reporting_days': fraud_indicators['late_reporting'],
                'documentation_score': fraud_indicators['doc_score'],
                'witness_present': fraud_indicators['witness'],
                'police_report_filed': fraud_indicators['police_report'],
                'previous_claims_count': fraud_indicators['prev_claims'],
                'policy_age_days': (claim_date - pd.to_datetime(customer['customer_since'])).days,
                
                # Target variable (hidden - for model training)
                'is_fraud': is_fraud
            }
            claims.append(claim)
        
        self.claims_df = pd.DataFrame(claims)
        return self.claims_df
    
    def _age_distribution(self):
        """Realistic age distribution for insurance customers."""
        ages = range(18, 85)
        weights = []
        for age in ages:
            if age < 25:
                w = 0.5
            elif age < 35:
                w = 1.2
            elif age < 50:
                w = 1.5
            elif age < 65:
                w = 1.3
            else:
                w = 0.8
            weights.append(w)
        total = sum(weights)
        return [w/total for w in weights]
    
    def _generate_auto_claim_amount(self, claim_type: str) -> float:
        """Generate realistic auto claim amounts."""
        amounts = {
            'Collision': np.random.lognormal(8.5, 0.8),
            'Comprehensive': np.random.lognormal(7.5, 0.9),
            'Liability': np.random.lognormal(9.0, 1.0),
            'Theft': np.random.lognormal(9.5, 0.5),
            'Vandalism': np.random.lognormal(6.5, 0.7)
        }
        return min(amounts.get(claim_type, 5000), 150000)
    
    def _generate_home_claim_amount(self, claim_type: str) -> float:
        """Generate realistic home claim amounts."""
        amounts = {
            'Water Damage': np.random.lognormal(8.8, 0.9),
            'Fire': np.random.lognormal(10.5, 1.2),
            'Theft': np.random.lognormal(8.0, 0.8),
            'Weather': np.random.lognormal(8.5, 1.0),
            'Liability': np.random.lognormal(9.0, 1.1),
            'Vandalism': np.random.lognormal(7.0, 0.6)
        }
        return min(amounts.get(claim_type, 10000), 500000)
    
    def _generate_claim_date(self, start_date: datetime, claim_type: str) -> datetime:
        """Generate claim date with seasonality."""
        days_offset = np.random.randint(0, 730)
        claim_date = start_date + timedelta(days=days_offset)
        
        # Weather claims more likely in winter/storm season - bump forward if needed
        if claim_type in ['Weather', 'Collision']:
            month = claim_date.month
            if month not in [12, 1, 2, 3] and np.random.random() < 0.3:
                # Move to a winter month by adding days
                days_to_winter = (12 - month) * 30 if month < 12 else 0
                claim_date = claim_date + timedelta(days=days_to_winter)
        
        return claim_date
    
    def _apply_fraud_patterns(self, base_amount: float, claim_type: str, 
                              customer: pd.Series) -> tuple:
        """Apply realistic fraud patterns to claim."""
        fraud_type = np.random.choice([
            'inflated_amount', 'staged_claim', 'repeat_offender', 'soft_fraud'
        ], p=[0.35, 0.25, 0.20, 0.20])
        
        indicators = {}
        
        if fraud_type == 'inflated_amount':
            amount = base_amount * np.random.uniform(1.8, 3.5)
            indicators['amount_ratio'] = np.random.uniform(2.0, 4.0)
            indicators['doc_score'] = np.random.randint(40, 70)
        elif fraud_type == 'staged_claim':
            amount = base_amount * np.random.uniform(1.3, 2.0)
            indicators['witness'] = 0
            indicators['police_report'] = np.random.choice([0, 1], p=[0.7, 0.3])
            indicators['late_reporting'] = np.random.randint(5, 20)
        elif fraud_type == 'repeat_offender':
            amount = base_amount * np.random.uniform(1.2, 1.8)
            indicators['claim_frequency'] = np.random.randint(3, 8)
            indicators['prev_claims'] = np.random.randint(4, 12)
        else:  # soft_fraud
            amount = base_amount * np.random.uniform(1.1, 1.4)
            indicators['doc_score'] = np.random.randint(50, 75)
        
        # Fill in remaining indicators
        indicators.setdefault('claim_frequency', np.random.randint(0, 4))
        indicators.setdefault('amount_ratio', np.random.uniform(1.2, 2.5))
        indicators.setdefault('weekend_filing', np.random.choice([0, 1], p=[0.4, 0.6]))
        indicators.setdefault('late_reporting', np.random.randint(2, 15))
        indicators.setdefault('doc_score', np.random.randint(50, 80))
        indicators.setdefault('witness', np.random.choice([0, 1], p=[0.6, 0.4]))
        indicators.setdefault('police_report', np.random.choice([0, 1], p=[0.5, 0.5]))
        indicators.setdefault('prev_claims', np.random.randint(2, 8))
        
        return amount, indicators
    
    def _generate_normal_indicators(self) -> dict:
        """Generate normal (non-fraud) indicators."""
        return {
            'claim_frequency': np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]),
            'amount_ratio': np.random.uniform(0.7, 1.3),
            'weekend_filing': np.random.choice([0, 1], p=[0.7, 0.3]),
            'late_reporting': np.random.randint(0, 5),
            'doc_score': np.random.randint(75, 100),
            'witness': np.random.choice([0, 1], p=[0.3, 0.7]),
            'police_report': np.random.choice([0, 1], p=[0.4, 0.6]),
            'prev_claims': np.random.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.15, 0.1])
        }
    
    def _get_deductible(self, policy_type: str, premium_tier: str) -> float:
        """Get deductible based on policy and tier."""
        deductibles = {
            'Auto': {'Bronze': 1000, 'Silver': 750, 'Gold': 500, 'Platinum': 250},
            'Home': {'Bronze': 2000, 'Silver': 1500, 'Gold': 1000, 'Platinum': 500}
        }
        return deductibles.get(policy_type, {}).get(premium_tier, 500)
    
    def export_to_csv(self, output_dir: str = 'data'):
        """Export all datasets to CSV for Power BI."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.customers_df is None:
            self.generate_customers()
        if self.claims_df is None:
            self.generate_claims()
        
        # Export customers
        self.customers_df.to_csv(f'{output_dir}/customers.csv', index=False)
        print(f"✓ Exported {len(self.customers_df)} customers to {output_dir}/customers.csv")
        
        # Export claims (without is_fraud for Power BI - that's for internal ML)
        claims_export = self.claims_df.drop(columns=['is_fraud'])
        claims_export.to_csv(f'{output_dir}/claims_master.csv', index=False)
        print(f"✓ Exported {len(claims_export)} claims to {output_dir}/claims_master.csv")
        
        # Create date dimension
        self._create_date_dimension(output_dir)
        
        return self.customers_df, self.claims_df
    
    def _create_date_dimension(self, output_dir: str):
        """Create date dimension table."""
        dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
        date_dim = pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'quarter': dates.quarter,
            'month': dates.month,
            'month_name': dates.strftime('%B'),
            'week': dates.isocalendar().week,
            'day_of_week': dates.dayofweek,
            'day_name': dates.strftime('%A'),
            'is_weekend': dates.dayofweek >= 5
        })
        date_dim.to_csv(f'{output_dir}/date_dimension.csv', index=False)
        print(f"✓ Exported date dimension to {output_dir}/date_dimension.csv")


if __name__ == '__main__':
    generator = InsuranceDataGenerator(n_customers=5000, n_claims=12000)
    customers, claims = generator.export_to_csv()
    print(f"\nData generation complete!")
    print(f"Fraud rate: {claims['is_fraud'].mean()*100:.1f}%")
