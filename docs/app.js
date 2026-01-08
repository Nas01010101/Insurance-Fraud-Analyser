// Chart.js config
Chart.defaults.color = '#666666';
Chart.defaults.borderColor = '#e5e5e5';
Chart.defaults.font.family = "'Inter', sans-serif";

const colors = {
    primary: '#0066cc',
    secondary: '#6366f1',
    success: '#16a34a',
    warning: '#ea580c',
    danger: '#dc2626',
    info: '#0891b2'
};

const formatCurrency = (v) => v >= 1e6 ? '$' + (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? '$' + (v / 1e3).toFixed(0) + 'K' : '$' + v;
const formatNumber = (v) => v.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");

async function init() {
    try {
        const res = await fetch('data/dashboard.json');
        const data = await res.json();
        render(data);
    } catch (e) {
        console.error(e);
        render(getDemoData());
    }
}

function render(data) {
    // KPIs
    document.getElementById('kpi-claims').textContent = formatNumber(data.overview.totalClaims);
    document.getElementById('kpi-payout').textContent = formatCurrency(data.overview.totalPayout);
    document.getElementById('kpi-approval').textContent = data.overview.approvalRate + '%';
    document.getElementById('kpi-processing').textContent = data.overview.avgProcessingDays;

    // Fraud stats
    document.getElementById('fraud-critical').textContent = formatNumber(data.riskDistribution.Critical || 0);
    document.getElementById('fraud-high').textContent = formatNumber(data.riskDistribution.High || 0);
    document.getElementById('fraud-medium').textContent = formatNumber(data.riskDistribution.Medium || 0);
    document.getElementById('fraud-low').textContent = formatNumber(data.riskDistribution.Low || 0);

    // Charts
    new Chart(document.getElementById('trendChart'), {
        type: 'line',
        data: {
            labels: data.monthlyTrend.map(d => d.month.slice(5)),
            datasets: [{
                data: data.monthlyTrend.map(d => d.count),
                borderColor: colors.primary,
                backgroundColor: 'rgba(0, 102, 204, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: { responsive: true, plugins: { legend: { display: false } } }
    });

    new Chart(document.getElementById('typeChart'), {
        type: 'doughnut',
        data: {
            labels: data.claimsByType.map(d => d.type),
            datasets: [{ data: data.claimsByType.map(d => d.count), backgroundColor: [colors.primary, colors.secondary, colors.info, colors.warning, colors.success] }]
        },
        options: { responsive: true, plugins: { legend: { position: 'right' } } }
    });

    new Chart(document.getElementById('provinceChart'), {
        type: 'bar',
        data: {
            labels: data.provinceData.map(d => d.province),
            datasets: [{ data: data.provinceData.map(d => d.count), backgroundColor: colors.primary }]
        },
        options: { responsive: true, plugins: { legend: { display: false } } }
    });

    new Chart(document.getElementById('statusChart'), {
        type: 'doughnut',
        data: {
            labels: Object.keys(data.statusDistribution),
            datasets: [{ data: Object.values(data.statusDistribution), backgroundColor: [colors.success, colors.danger, colors.warning, colors.info] }]
        },
        options: { responsive: true, plugins: { legend: { position: 'right' } } }
    });

    new Chart(document.getElementById('riskChart'), {
        type: 'doughnut',
        data: {
            labels: ['Low', 'Medium', 'High', 'Critical'],
            datasets: [{ data: ['Low', 'Medium', 'High', 'Critical'].map(k => data.riskDistribution[k] || 0), backgroundColor: [colors.success, colors.info, colors.warning, colors.danger] }]
        },
        options: { responsive: true, plugins: { legend: { position: 'right' } } }
    });

    new Chart(document.getElementById('importanceChart'), {
        type: 'bar',
        data: {
            labels: data.featureImportance.map(d => d.feature.replace(/_/g, ' ')),
            datasets: [{ data: data.featureImportance.map(d => (d.importance * 100).toFixed(1)), backgroundColor: colors.secondary }]
        },
        options: { indexAxis: 'y', responsive: true, plugins: { legend: { display: false } }, scales: { x: { max: 35 } } }
    });

    // Table
    document.getElementById('claims-table').innerHTML = data.highRiskClaims.map(c => `
        <tr>
            <td><code>${c.claim_id}</code></td>
            <td>${c.claim_type}</td>
            <td>${formatCurrency(c.claim_amount)}</td>
            <td>${c.risk_score.toFixed(1)}</td>
            <td><span class="risk-badge ${c.risk_category}">${c.risk_category}</span></td>
        </tr>
    `).join('');
}

function getDemoData() {
    return {
        overview: { totalClaims: 12000, totalPayout: 85420000, approvalRate: 65.2, avgProcessingDays: 15.3 },
        riskDistribution: { Low: 11114, Medium: 77, High: 13, Critical: 795 },
        monthlyTrend: [
            { month: '2024-01', count: 980 }, { month: '2024-02', count: 920 }, { month: '2024-03', count: 1050 },
            { month: '2024-04', count: 1100 }, { month: '2024-05', count: 980 }, { month: '2024-06', count: 1020 }
        ],
        claimsByType: [
            { type: 'Collision', count: 4800 }, { type: 'Comprehensive', count: 3000 },
            { type: 'Water Damage', count: 2100 }, { type: 'Theft', count: 1200 }
        ],
        policySplit: { Auto: 7800, Home: 4200 },
        statusDistribution: { Approved: 7800, Denied: 1800, 'Under Review': 1200, Closed: 1200 },
        featureImportance: [
            { feature: 'amount_vs_avg_ratio', importance: 0.287 },
            { feature: 'documentation_score', importance: 0.200 },
            { feature: 'late_reporting_days', importance: 0.146 }
        ],
        provinceData: [
            { province: 'ON', count: 4560 }, { province: 'QC', count: 2760 },
            { province: 'BC', count: 1560 }, { province: 'AB', count: 1440 }
        ],
        highRiskClaims: [
            { claim_id: 'CLM0001234', claim_type: 'Collision', claim_amount: 45000, risk_score: 92, risk_category: 'Critical' },
            { claim_id: 'CLM0002567', claim_type: 'Theft', claim_amount: 38000, risk_score: 88, risk_category: 'Critical' }
        ]
    };
}

document.addEventListener('DOMContentLoaded', init);
