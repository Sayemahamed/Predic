const vscode = acquireVsCodeApi();

// Elements
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const attachButton = document.getElementById('attach-button');
const contextBar = document.getElementById('context-bar');
const typingIndicator = document.getElementById('typing-indicator');

// Icons (Lucide Style)
const ICONS = {
    copy: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
    check: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
    insert: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>`,
    close: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
    file: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>`
};

// --- Handlers ---
window.addEventListener('message', event => {
    const msg = event.data;
    switch (msg.type) {
        case 'addMessage':
            // Remove welcome message on first chat
            const welcome = document.querySelector('.welcome-message');
            if (welcome) welcome.remove();
            addMessageToUI(msg.role, msg.content);
            break;
        case 'fileAttached':
            addContextChip(msg.fileName);
            break;
        case 'clearContext':
            contextBar.innerHTML = '';
            contextBar.classList.add('hidden');
            break;
        case 'setLoading':
            if (msg.value) typingIndicator.classList.remove('hidden');
            else typingIndicator.classList.add('hidden');
            break;
    }
});

function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;
    
    vscode.postMessage({ type: 'sendMessage', value: text });
    messageInput.value = '';
    messageInput.style.height = 'auto'; // Reset height
}

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

attachButton.addEventListener('click', () => {
    vscode.postMessage({ type: 'attachFile' });
});

sendButton.addEventListener('click', sendMessage);

// --- UI Builders ---

function addContextChip(name) {
    contextBar.classList.remove('hidden');
    const chip = document.createElement('div');
    chip.className = 'context-chip';
    chip.innerHTML = `${ICONS.file} <span>${name}</span>`;
    contextBar.appendChild(chip);
}

function addMessageToUI(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    
    // Parse Code Blocks
    const parts = content.split(/```(\w*)\n([\s\S]*?)```/g);
    
    parts.forEach((part, index) => {
        if (index % 3 === 0) {
            if (part.trim()) {
                const p = document.createElement('div');
                p.innerHTML = formatMarkdown(part); // Simple bold/italic
                div.appendChild(p);
            }
        } else if (index % 3 === 2) {
            const lang = parts[index-1] || 'text';
            div.appendChild(createCodeBlock(part, lang));
        }
    });

    // Fallback for plain text
    if (parts.length === 1) {
        div.innerHTML = formatMarkdown(content);
    }

    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function createCodeBlock(code, lang) {
    const container = document.createElement('div');
    container.className = 'code-block-container';

    const header = document.createElement('div');
    header.className = 'code-header';
    header.innerHTML = `<span class="lang-label">${lang}</span>`;

    const actions = document.createElement('div');
    actions.className = 'code-actions';

    // Copy Button
    const copyBtn = createBtn('Copy', ICONS.copy, () => {
        navigator.clipboard.writeText(code);
        copyBtn.innerHTML = `${ICONS.check} <span>Copied</span>`;
        setTimeout(() => copyBtn.innerHTML = `${ICONS.copy} <span>Copy</span>`, 2000);
    });

    // Insert Button
    const insertBtn = createBtn('Insert', ICONS.insert, () => {
        vscode.postMessage({ type: 'insertCode', value: code });
    });

    actions.append(copyBtn, insertBtn);
    header.appendChild(actions);

    const pre = document.createElement('pre');
    const codeEl = document.createElement('code');
    codeEl.textContent = code;
    pre.appendChild(codeEl);

    container.append(header, pre);
    return container;
}

function createBtn(label, icon, onClick) {
    const btn = document.createElement('button');
    btn.className = 'action-btn';
    btn.title = label;
    btn.innerHTML = `${icon} <span>${label}</span>`; // Text label for clarity
    btn.addEventListener('click', onClick);
    return btn;
}

function formatMarkdown(text) {
    // Basic formatting for bold and inline code
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>');
}