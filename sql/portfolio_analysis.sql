-- =====================================================
-- Insurance Claims Portfolio Analysis SQL Queries
-- For TD Insurance Data Analytics Project
-- =====================================================

-- ===================
-- CLAIMS OVERVIEW
-- ===================

-- Total claims by policy type
SELECT 
    policy_type,
    COUNT(*) as total_claims,
    SUM(claim_amount) as total_amount,
    AVG(claim_amount) as avg_amount,
    SUM(payout_amount) as total_payout
FROM claims_master
GROUP BY policy_type
ORDER BY total_claims DESC;

-- Monthly claims trend
SELECT 
    SUBSTR(claim_date, 1, 7) as month,
    COUNT(*) as claim_count,
    SUM(claim_amount) as total_amount,
    AVG(claim_amount) as avg_amount
FROM claims_master
GROUP BY SUBSTR(claim_date, 1, 7)
ORDER BY month;

-- Claims by type and status
SELECT 
    claim_type,
    claim_status,
    COUNT(*) as count,
    SUM(claim_amount) as total_amount
FROM claims_master
GROUP BY claim_type, claim_status
ORDER BY claim_type, count DESC;


-- ===================
-- GEOGRAPHIC ANALYSIS
-- ===================

-- Claims distribution by province
SELECT 
    province,
    COUNT(*) as claim_count,
    SUM(claim_amount) as total_claims,
    AVG(claim_amount) as avg_claim,
    SUM(payout_amount) as total_payout
FROM claims_master
GROUP BY province
ORDER BY claim_count DESC;

-- Loss ratio by province (payout / premium equivalent)
SELECT 
    province,
    SUM(payout_amount) as total_payout,
    COUNT(*) as claims,
    SUM(payout_amount) / COUNT(*) as avg_payout_per_claim
FROM claims_master
WHERE claim_status = 'Approved'
GROUP BY province
ORDER BY avg_payout_per_claim DESC;


-- ===================
-- PERFORMANCE METRICS
-- ===================

-- Claims processing time analysis
SELECT 
    claim_status,
    AVG(processing_days) as avg_processing_days,
    MIN(processing_days) as min_days,
    MAX(processing_days) as max_days,
    COUNT(*) as claim_count
FROM claims_master
GROUP BY claim_status;

-- Adjuster workload and performance
SELECT 
    adjuster_id,
    COUNT(*) as claims_handled,
    AVG(claim_amount) as avg_claim_amount,
    AVG(processing_days) as avg_processing_time,
    SUM(CASE WHEN claim_status = 'Approved' THEN 1 ELSE 0 END) as approved_count,
    SUM(CASE WHEN claim_status = 'Denied' THEN 1 ELSE 0 END) as denied_count
FROM claims_master
GROUP BY adjuster_id
ORDER BY claims_handled DESC
LIMIT 20;


-- ===================
-- CUSTOMER INSIGHTS
-- ===================

-- Customer claim frequency analysis
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name as customer_name,
    c.premium_tier,
    COUNT(cl.claim_id) as total_claims,
    SUM(cl.claim_amount) as total_claimed,
    AVG(cl.claim_amount) as avg_claim
FROM customers c
LEFT JOIN claims_master cl ON c.customer_id = cl.customer_id
GROUP BY c.customer_id, customer_name, c.premium_tier
HAVING COUNT(cl.claim_id) > 0
ORDER BY total_claims DESC
LIMIT 50;

-- Claims by customer age group
SELECT 
    CASE 
        WHEN c.age < 25 THEN '18-24'
        WHEN c.age < 35 THEN '25-34'
        WHEN c.age < 45 THEN '35-44'
        WHEN c.age < 55 THEN '45-54'
        WHEN c.age < 65 THEN '55-64'
        ELSE '65+'
    END as age_group,
    COUNT(cl.claim_id) as claim_count,
    AVG(cl.claim_amount) as avg_claim
FROM customers c
JOIN claims_master cl ON c.customer_id = cl.customer_id
GROUP BY age_group
ORDER BY age_group;

-- Premium tier performance
SELECT 
    c.premium_tier,
    COUNT(DISTINCT c.customer_id) as customers,
    COUNT(cl.claim_id) as claims,
    SUM(cl.claim_amount) as total_claimed,
    SUM(cl.payout_amount) as total_payout,
    CAST(COUNT(cl.claim_id) AS FLOAT) / COUNT(DISTINCT c.customer_id) as claims_per_customer
FROM customers c
LEFT JOIN claims_master cl ON c.customer_id = cl.customer_id
GROUP BY c.premium_tier
ORDER BY c.premium_tier;


-- ===================
-- SEASONAL PATTERNS
-- ===================

-- Claims by day of week
SELECT 
    CASE CAST(strftime('%w', claim_date) AS INTEGER)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END as day_of_week,
    COUNT(*) as claim_count,
    AVG(claim_amount) as avg_amount
FROM claims_master
GROUP BY strftime('%w', claim_date)
ORDER BY CAST(strftime('%w', claim_date) AS INTEGER);

-- Quarterly trends
SELECT 
    SUBSTR(claim_date, 1, 4) as year,
    CASE 
        WHEN CAST(SUBSTR(claim_date, 6, 2) AS INTEGER) <= 3 THEN 'Q1'
        WHEN CAST(SUBSTR(claim_date, 6, 2) AS INTEGER) <= 6 THEN 'Q2'
        WHEN CAST(SUBSTR(claim_date, 6, 2) AS INTEGER) <= 9 THEN 'Q3'
        ELSE 'Q4'
    END as quarter,
    claim_type,
    COUNT(*) as claims,
    SUM(claim_amount) as total_amount
FROM claims_master
GROUP BY year, quarter, claim_type
ORDER BY year, quarter, claim_type;
