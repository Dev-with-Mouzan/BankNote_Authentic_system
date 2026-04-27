document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const form = e.target;
    const predictBtn = document.getElementById('predictBtn');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    const predictionDiv = document.getElementById('prediction');
    const probabilityDiv = document.getElementById('probability');
    
    const data = {
        variance: parseFloat(form.variance.value),
        skewness: parseFloat(form.skewness.value),
        curtosis: parseFloat(form.curtosis.value),
        entropy: parseFloat(form.entropy.value)
    };
    
    resultDiv.classList.add('hidden');
    resultDiv.classList.remove('legitimate', 'fake');
    predictBtn.disabled = true;
    loadingDiv.classList.remove('hidden');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Network response was not ok');
        }
        
        const result = await response.json();
        
        loadingDiv.classList.add('hidden');
        
        predictionDiv.textContent = result.message;
        probabilityDiv.textContent = `Confidence: ${(result.probability * 100).toFixed(2)}%`;
        
        resultDiv.classList.remove('hidden');
        
        if (result.prediction === 0) {
            resultDiv.classList.add('legitimate');
        } else {
            resultDiv.classList.add('fake');
        }
    } catch (error) {
        loadingDiv.classList.add('hidden');
        alert('Error: ' + error.message);
    }
    
    predictBtn.disabled = false;
});