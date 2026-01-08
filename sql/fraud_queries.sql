-- =====================================================
-- Fraud Detection & Anomaly Analysis SQL Queries
-- For TD Insurance Data Analytics Project
-- =====================================================

-- ===================
-- HIGH-RISK CLAIMS
-- ===================

-- Claims with high fraud risk scores
SELECT 
    cm.claim_id,
    cm.customer_id,
    cm.claim_type,
    cm.claim_amount,
    cm.claim_date,
    fs.risk_score,
    fs.risk_category,
    cm.documentation_score,
    cm.police_report_filed,
    cm.witness_present
FROM claims_master cm
JOIN fraud_scores fs ON cm.claim_id = fs.claim_id
WHERE fs.risk_score >= 70
ORDER BY fs.risk_score DESC;

-- Critical risk claims requiring immediate investigation
SELECT 
    cm.*,
    fs.risk_score,
    fs.anomaly_score,
    fs.ml_fraud_probability
FROM claims_master cm
JOIN fraud_scores fs ON cm.claim_id = fs.claim_id
WHERE fs.risk_category = 'Critical'
ORDER BY fs.risk_score DESC;


-- ===================
-- FRAUD PATTERN ANALYSIS
-- ===================

-- High-value claims with suspicious indicators
SELECT 
    claim_id,
    claim_type,
    claim_amount,
    documentation_score,
    late_reporting_days,
    witness_present,
    police_report_filed,
    claim_frequency_90d as recent_claims
FROM claims_master
WHERE claim_amount > (SELECT AVG(claim_amount) * 2 FROM claims_master)
  AND (documentation_score < 70 
       OR late_reporting_days > 7 
       OR (witness_present = 0 AND police_report_filed = 0))
ORDER BY claim_amount DESC;

-- Repeat claimers (potential fraud pattern)
SELECT 
    customer_id,
    COUNT(*) as claim_count,
    SUM(claim_amount) as total_claimed,
    AVG(documentation_score) as avg_doc_score,
    MAX(claim_frequency_90d) as max_recent_frequency
FROM claims_master
GROUP BY customer_id
HAVING COUNT(*) >= 5
ORDER BY claim_count DESC;

-- Weekend filing anomaly (fraud often filed on weekends)
SELECT 
    CASE WHEN weekend_filing = 1 THEN 'Weekend' ELSE 'Weekday' END as filing_day,
    COUNT(*) as claim_count,
    AVG(claim_amount) as avg_amount,
    AVG(documentation_score) as avg_doc_score,
    SUM(CASE WHEN police_report_filed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as police_report_pct
FROM claims_master
GROUP BY weekend_filing;


-- ===================
-- FRAUD BY CATEGORY
-- ===================

-- Fraud risk distribution by claim type
SELECT 
    cm.claim_type,
    COUNT(*) as total_claims,
    SUM(CASE WHEN fs.risk_category = 'Critical' THEN 1 ELSE 0 END) as critical_count,
    SUM(CASE WHEN fs.risk_category = 'High' THEN 1 ELSE 0 END) as high_count,
    AVG(fs.risk_score) as avg_risk_score,
    SUM(CASE WHEN fs.flagged_for_review = 1 THEN 1 ELSE 0 END) as flagged_count
FROM claims_master cm
JOIN fraud_scores fs ON cm.claim_id = fs.claim_id
GROUP BY cm.claim_type
ORDER BY avg_risk_score DESC;

-- Fraud indicators by province
SELECT 
    province,
    COUNT(*) as claims,
    AVG(documentation_score) as avg_doc_score,
    AVG(late_reporting_days) as avg_late_days,
    SUM(CASE WHEN police_report_filed = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as no_police_pct,
    SUM(CASE WHEN witness_present = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as no_witness_pct
FROM claims_master
GROUP BY province
ORDER BY avg_doc_score;


-- ===================
-- ANOMALY DETECTION
-- ===================

-- Amount outliers by claim type
SELECT 
    claim_type,
    claim_id,
    claim_amount,
    (SELECT AVG(claim_amount) FROM claims_master cm2 WHERE cm2.claim_type = cm.claim_type) as type_avg,
    claim_amount / (SELECT AVG(claim_amount) FROM claims_master cm2 WHERE cm2.claim_type = cm.claim_type) as ratio_to_avg
FROM claims_master cm
WHERE claim_amount > (
    SELECT AVG(claim_amount) + 2 * 
           (SELECT AVG(claim_amount * claim_amount) - AVG(claim_amount) * AVG(claim_amount) FROM claims_master cm3 WHERE cm3.claim_type = cm.claim_type)
    FROM claims_master cm2 
    WHERE cm2.claim_type = cm.claim_type
)
ORDER BY ratio_to_avg DESC
LIMIT 100;

-- New customers with immediate high claims (soft fraud indicator)
SELECT 
    cm.claim_id,
    cm.customer_id,
    cm.claim_amount,
    cm.policy_age_days,
    c.customer_since,
    fs.risk_score
FROM claims_master cm
JOIN customers c ON cm.customer_id = c.customer_id
JOIN fraud_scores fs ON cm.claim_id = fs.claim_id
WHERE cm.policy_age_days < 90
  AND cm.claim_amount > 10000
ORDER BY cm.claim_amount DESC;


-- ===================
-- FRAUD SUMMARY METRICS
-- ===================

-- Overall fraud risk summary
SELECT 
    risk_category,
    COUNT(*) as claims,
    SUM(CASE WHEN flagged_for_review = 1 THEN 1 ELSE 0 END) as flagged,
    SUM(CASE WHEN requires_investigation = 1 THEN 1 ELSE 0 END) as investigation_needed
FROM fraud_scores
GROUP BY risk_category
ORDER BY 
    CASE risk_category 
        WHEN 'Critical' THEN 1 
        WHEN 'High' THEN 2 
        WHEN 'Medium' THEN 3 
        ELSE 4 
    END;

-- Monthly fraud trend
SELECT 
    SUBSTR(cm.claim_date, 1, 7) as month,
    COUNT(*) as total_claims,
    SUM(CASE WHEN fs.risk_category IN ('High', 'Critical') THEN 1 ELSE 0 END) as high_risk_claims,
    AVG(fs.risk_score) as avg_risk_score,
    SUM(CASE WHEN fs.flagged_for_review = 1 THEN cm.claim_amount ELSE 0 END) as flagged_amount
FROM claims_master cm
JOIN fraud_scores fs ON cm.claim_id = fs.claim_id
GROUP BY SUBSTR(cm.claim_date, 1, 7)
ORDER BY month;
