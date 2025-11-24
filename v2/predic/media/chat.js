const vscode = acquireVsCodeApi();

const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const attachButton = document.getElementById('attach-button');
const contextIndicator = document.getElementById('context-indicator');
const contextFilename = document.getElementById('context-filename');

// --- SVGs for Buttons (Reliable!) ---
const ICONS = {
    copy: `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M4 4H12V12H4V4Z M3 3V13H13V3H3Z M1 1V10H2V2H10V1H1Z"/></svg>`,
    check: `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/></svg>`,
    close: `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/></svg>`
};

window.addEventListener('message', event => {
    const message = event.data;
    switch (message.type) {
        case 'addMessage':
            addMessageToUI(message.role, message.content);
            break;
        case 'fileAttached':
            showContext(message.fileName);
            break;
        case 'clearContext':
            hideContext();
            break;
    }
});

function sendMessage() {
    const text = messageInput.value.trim();
    if (text) {
        vscode.postMessage({ type: 'sendMessage', value: text });
        messageInput.value = '';
    }
}

attachButton.addEventListener('click', () => {
    vscode.postMessage({ type: 'attachFile' });
});

function showContext(name) {
    contextFilename.textContent = name;
    contextIndicator.classList.remove('hidden');
}

function hideContext() {
    contextIndicator.classList.add('hidden');
    contextFilename.textContent = '';
}

sendButton.addEventListener('click', sendMessage);

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

function addMessageToUI(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const parts = content.split(/```(\w*)\n([\s\S]*?)```/g);
    
    for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        if (i % 3 === 0) {
            if (part.trim()) {
                const p = document.createElement('div');
                p.className = 'markdown-text';
                p.innerText = part; 
                messageDiv.appendChild(p);
            }
        } else if (i % 3 === 1) {
            // language capture
        } else {
            const lang = parts[i-1] || 'text';
            const codeBlock = createCodeBlock(part, lang);
            messageDiv.appendChild(codeBlock);
        }
    }
    
    if (parts.length === 1) messageDiv.innerText = content;

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function createCodeBlock(code, lang) {
    const container = document.createElement('div');
    container.className = 'code-block-container';

    const header = document.createElement('div');
    header.className = 'code-header';
    
    const langLabel = document.createElement('span');
    langLabel.className = 'lang-label';
    langLabel.textContent = lang;

    const actions = document.createElement('div');
    actions.className = 'code-actions';

    const btnCopy = createActionButton('copy', 'Copy', () => {
        navigator.clipboard.writeText(code);
        btnCopy.innerHTML = ICONS.check;
        setTimeout(() => btnCopy.innerHTML = ICONS.copy, 2000);
    });

    const btnAccept = createActionButton('check', 'Accept', () => {
        vscode.postMessage({ type: 'insertCode', value: code });
    });

    const btnReject = createActionButton('close', 'Reject', () => {
        container.remove();
    });

    actions.append(btnCopy, btnAccept, btnReject);
    header.append(langLabel, actions);

    const pre = document.createElement('pre');
    const codeEl = document.createElement('code');
    codeEl.textContent = code;
    pre.appendChild(codeEl);

    container.append(header, pre);
    return container;
}

function createActionButton(iconKey, title, onClick) {
    const btn = document.createElement('button');
    btn.className = 'code-action-btn';
    btn.title = title;
    btn.innerHTML = ICONS[iconKey];
    btn.addEventListener('click', onClick);
    return btn;
}