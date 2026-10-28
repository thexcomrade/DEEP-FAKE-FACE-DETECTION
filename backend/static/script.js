/**
 * Deepfake Face Detection Frontend Utilities
 */

function checkApiStatus() {
    fetch('/api/status')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('System Status:', data);
            const output = document.getElementById("output");
            if (output) {
                output.innerText = `Status: ${data.status} | Model: ${data.model_type || 'offline'}`;
            }
        })
        .catch(error => console.error('Status check error:', error));
}
