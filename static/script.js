function fetchData() {
    fetch('/data')
        .then(response => response.json())
        .then(data => {
            document.getElementById("output").innerText = data.message;
        })
        .catch(error => console.error('Error:', error));
}
