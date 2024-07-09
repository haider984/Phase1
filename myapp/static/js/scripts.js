let file = null;

function uploadDocument() {
    const fileInput = document.getElementById('file-input');
    file = fileInput.files[0];

    if (file) {
        document.getElementById('upload-section').style.display = 'none';
        document.getElementById('chat-section').style.display = 'block';
    } else {
        alert('Please select a file first.');
    }
}

function askQuestion() {
    const question = document.getElementById('question-input').value;
    if (!question) {
        alert('Please enter a question.');
        return;
    }

    if (!file) {
        alert('No document uploaded.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('query', question);

    fetch('/api/query-document/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const chatbox = document.getElementById('chatbox');
        chatbox.innerHTML += `<div><strong>You:</strong> ${question}</div>`;
        chatbox.innerHTML += `<div><strong>Bot:</strong> ${data.response}</div>`;
        document.getElementById('question-input').value = '';
        chatbox.scrollTop = chatbox.scrollHeight;
    })
    .catch(error => console.error('Error:', error));
}
