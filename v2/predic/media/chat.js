(function() {
    const vscode = acquireVsCodeApi();
    const messagesContainer = document.getElementById('messages');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearButton');
    const modelSelector = document.getElementById('modelSelector');
    const refreshModelsBtn = document.getElementById('refreshModelsBtn');

    let messages = [];
    let models = [];
    let selectedModel = '';
    let isTyping = false;

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    clearButton.addEventListener('click', clearChat);
    modelSelector.addEventListener('change', selectModel);
    refreshModelsBtn.addEventListener('click', refreshModels);
    
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener('input', () => {
        sendButton.disabled = !messageInput.value.trim() || !selectedModel;
    });

    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message || !selectedModel) return;

        vscode.postMessage({
            type: 'sendMessage',
            message: message
        });

        messageInput.value = '';
        sendButton.disabled = true;
        showTypingIndicator();
    }

    function clearChat() {
        if (messages.length > 0 && confirm('Are you sure you want to clear the chat history?')) {
            vscode.postMessage({
                type: 'clear'
            });
        }
    }

    function selectModel() {
        const modelId = modelSelector.value;
        if (modelId) {
            vscode.postMessage({
                type: 'selectModel',
                modelId: modelId
            });
        }
    }

    function refreshModels() {
        vscode.postMessage({
            type: 'refreshModels'
        });
    }

    function renderMessages() {
        if (messages.length === 0) {
            messagesContainer.innerHTML = `
                <div class="welcome-message">
                    <i class="codicon codicon-hubot large-icon"></i>
                    <h2>Welcome to Predic!</h2>
                    <p>Select a model above to start chatting with your AI coding assistant.</p>
                </div>
            `;
            return;
        }

        messagesContainer.innerHTML = '';
        
        messages.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.role}`;
            
            const icon = document.createElement('div');
            icon.className = 'message-icon';
            icon.innerHTML = msg.role === 'user' 
                ? '<i class="codicon codicon-account"></i>' 
                : '<i class="codicon codicon-hubot"></i>';
            
            const content = document.createElement('div');
            content.className = 'message-content';
            
            const header = document.createElement('div');
            header.className = 'message-header';
            header.innerHTML = `
                <span class="message-role">${msg.role === 'user' ? 'You' : 'Predic'}</span>
                <span class="message-time">${formatTime(msg.timestamp)}</span>
            `;
            
            const text = document.createElement('div');
            text.className = 'message-text';
            text.innerHTML = formatMessage(msg.content);
            
            content.appendChild(header);
            content.appendChild(text);
            
            messageDiv.appendChild(icon);
            messageDiv.appendChild(content);
            messagesContainer.appendChild(messageDiv);
        });

        if (isTyping) {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant';
            typingDiv.innerHTML = `
                <div class="message-icon">
                    <i class="codicon codicon-hubot"></i>
                </div>
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            `;
            messagesContainer.appendChild(typingDiv);
        }
        
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function showTypingIndicator() {
        isTyping = true;
        renderMessages();
    }

    function hideTypingIndicator() {
        isTyping = false;
        renderMessages();
    }

    function updateModelSelector() {
        modelSelector.innerHTML = '<option value="">Select a model...</option>';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            option.selected = model.id === selectedModel;
            modelSelector.appendChild(option);
        });

        sendButton.disabled = !messageInput.value.trim() || !selectedModel;
    }

    function formatTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function formatMessage(content) {
        // Simple markdown-like formatting
        return content
            .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    // Handle messages from extension
    window.addEventListener('message', event => {
        const message = event.data;
        switch (message.type) {
            case 'update':
                messages = message.messages || [];
                models = message.models || [];
                selectedModel = message.selectedModel || '';
                hideTypingIndicator();
                renderMessages();
                updateModelSelector();
                break;
        }
    });

    // Initial render
    renderMessages();
})();