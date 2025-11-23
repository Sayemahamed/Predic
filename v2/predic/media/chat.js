// Initialize VS Code API
const vscode = acquireVsCodeApi();

const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');

// 1. Handle Incoming Messages from Extension
window.addEventListener('message', event => {
    const message = event.data;
    switch (message.type) {
        case 'addMessage':
            addMessageToUI(message.role, message.content);
            break;
    }
});

// 2. Handle Sending
function sendMessage() {
    const text = messageInput.value.trim();
    if (text) {
        // Send to backend
        vscode.postMessage({ type: 'sendMessage', value: text });
        messageInput.value = '';
    }
}

sendButton.addEventListener('click', sendMessage);

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 3. UI Helper
function addMessageToUI(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    // Simple markdown-like code block handling
    if (content.includes('```')) {
        // Very basic formatting for code blocks
        const parts = content.split('```');
        parts.forEach((part, index) => {
            if (index % 2 === 1) { // Code block
                const code = document.createElement('pre');
                code.textContent = part;
                messageDiv.appendChild(code);
            } else { // Text
                const p = document.createElement('p');
                p.textContent = part;
                messageDiv.appendChild(p);
            }
        });
    } else {
        messageDiv.textContent = content;
    }

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}