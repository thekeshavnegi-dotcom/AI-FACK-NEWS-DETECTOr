const predictBtn = document.getElementById('predict-btn');
const newsText = document.getElementById('news-text');
const loader = document.getElementById('loader');
const resultCard = document.getElementById('result-card');
const resultLabel = document.getElementById('result-label');
const resultMessage = document.getElementById('result-message');
const resultConfidence = document.getElementById('result-confidence');

function setResult(data) {
    resultCard.classList.remove('hidden');
    resultLabel.textContent = data.label;
    resultMessage.textContent = data.message;
    resultConfidence.textContent = data.confidence;
    resultCard.style.borderColor = data.label === 'REAL' ? '#16a34a' : '#dc2626';
    resultCard.style.background = data.label === 'REAL'
        ? 'rgba(22, 163, 74, 0.12)'
        : 'rgba(220, 38, 38, 0.14)';
}

function showLoader(show) {
    loader.hidden = !show;
    predictBtn.disabled = show;
}

predictBtn.addEventListener('click', async () => {
    const text = newsText.value.trim();
    if (!text) {
        alert('Please enter a news article or text to analyze.');
        return;
    }

    showLoader(true);
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ news_text: text }),
    });
    showLoader(false);

    if (!response.ok) {
        const error = await response.json();
        alert(error.error || 'Prediction failed.');
        return;
    }

    const data = await response.json();
    setResult(data);
    if (window.predictionChart) {
        const fakeCount = initialCounts.fake + (data.label === 'FAKE' ? 1 : 0);
        const realCount = initialCounts.real + (data.label === 'REAL' ? 1 : 0);
        window.predictionChart.data.datasets[0].data = [fakeCount, realCount];
        window.predictionChart.update();
    }
});

function createChart() {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    window.predictionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Fake', 'Real'],
            datasets: [{
                data: [initialCounts.fake, initialCounts.real],
                backgroundColor: ['rgba(239, 68, 68, 0.8)', 'rgba(34, 197, 94, 0.8)'],
                borderColor: ['rgba(239, 68, 68, 0.4)', 'rgba(34, 197, 94, 0.4)'],
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#cbd5e1' } },
            },
        },
    });
}

createChart();
