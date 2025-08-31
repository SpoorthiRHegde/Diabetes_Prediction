document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('healthForm');
    const inputForm = document.getElementById('inputForm');
    const resultContainer = document.getElementById('resultContainer');
    const riskLevel = document.getElementById('riskLevel');
    const riskBar = document.getElementById('riskBar');
    const recommendationList = document.getElementById('recommendationList');
    const backBtn = document.getElementById('backBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = {};
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            formData[input.name] = input.value;
        });
        
        const predictBtn = document.getElementById('predictBtn');
        predictBtn.disabled = true;
        predictBtn.textContent = 'Analyzing...';
        
        try {
            // Make sure this points to your Flask server (usually port 5000)
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
                
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'success') {
                displayResults(data, formData);
            } else {
                throw new Error(data.error || 'Unknown error occurred');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error: ' + error.message);
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = 'Assess My Health Risk';
        }
    });
    
    function displayResults(data, formData) {
        inputForm.style.display = 'none';
        resultContainer.style.display = 'block';
        
        resultContainer.dataset.formData = JSON.stringify({
            ...formData, 
            prediction: data.prediction,
            probability: data.probability,
            timestamp: new Date().toISOString()
        });
        
        const riskPercentage = Math.round(data.probability * 100);
        riskBar.style.width = riskPercentage + '%';
        
        if (data.prediction === 1) {
            riskLevel.textContent = `High Risk (${riskPercentage}%)`;
            riskLevel.className = 'risk-level high-risk';
            riskBar.style.backgroundColor = '#dc3545'; // Red for high risk
            
            recommendationList.innerHTML = `
                <li>Consult with your healthcare provider about your risk factors</li>
                <li>Consider regular health screenings and check-ups</li>
                <li>Adopt a healthier diet with more fruits and vegetables</li>
                <li>Increase physical activity to at least 150 minutes per week</li>
                <li>If you smoke, consider quitting</li>
                <li>Monitor your blood pressure and cholesterol regularly</li>
                <li>Limit alcohol consumption</li>
                <li>Manage stress through relaxation techniques</li>
            `;
        } else {
            riskLevel.textContent = `Low Risk (${riskPercentage}%)`;
            riskLevel.className = 'risk-level low-risk';
            riskBar.style.backgroundColor = '#28a745'; // Green for low risk
            
            recommendationList.innerHTML = `
                <li>Continue with your healthy habits</li>
                <li>Maintain regular physical activity</li>
                <li>Keep up with balanced nutrition</li>
                <li>Schedule regular health check-ups</li>
                <li>Monitor any changes in your health status</li>
                <li>Stay aware of your family health history</li>
            `;
        }
        
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    backBtn.addEventListener('click', function() {
        resultContainer.style.display = 'none';
        inputForm.style.display = 'block';
        inputForm.scrollIntoView({ behavior: 'smooth' });
    });
    
    downloadBtn.addEventListener('click', async function() {
        const formData = JSON.parse(resultContainer.dataset.formData);
        
        // Update button state
        const originalText = downloadBtn.textContent;
        downloadBtn.disabled = true;
        downloadBtn.textContent = 'Generating PDF...';
        
        try {
            const response = await fetch('http://localhost:5000/generate_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            
            // Get the PDF blob
            const blob = await response.blob();
            
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'health_risk_assessment_report.pdf';
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            // Show success message
            showNotification('PDF report downloaded successfully!', 'success');
            
        } catch (error) {
            console.error('Error:', error);
            showNotification('Error downloading PDF report: ' + error.message, 'error');
        } finally {
            downloadBtn.disabled = false;
            downloadBtn.textContent = originalText;
        }
    });
    
    // Utility function to show notifications
    function showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '15px 20px',
            borderRadius: '5px',
            color: 'white',
            fontWeight: 'bold',
            zIndex: '1000',
            maxWidth: '300px',
            backgroundColor: type === 'success' ? '#28a745' : '#dc3545'
        });
        
        // Add to page
        document.body.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 5000);
    }
});

function openBMIPopup() {
    document.getElementById('bmiPopup').style.display = 'block';
    document.getElementById('bmiOverlay').style.display = 'block';
}

function closeBMIPopup() {
    document.getElementById('bmiPopup').style.display = 'none';
    document.getElementById('bmiOverlay').style.display = 'none';
    document.getElementById('height').value = '';
    document.getElementById('weight').value = '';
}

function calculateBMI() {
    const height = parseFloat(document.getElementById('height').value);
    const weight = parseFloat(document.getElementById('weight').value);

    if (!height || !weight || height <= 0 || weight <= 0) {
        alert("Please enter valid height and weight.");
        return;
    }

    const bmi = weight / (height * height);
    document.getElementById('bmi').value = bmi.toFixed(1);
    closeBMIPopup();
}
