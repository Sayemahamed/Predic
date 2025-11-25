const vscode = acquireVsCodeApi();

// Elements
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const attachButton = document.getElementById('attach-button');
const contextBar = document.getElementById('context-bar');
const typingIndicator = document.getElementById('typing-indicator');

// State
let isStreaming = false;
let activeMessageDiv = null;
let activeContent = ""; 

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
            const welcome = document.querySelector('.welcome-message');
            if (welcome) welcome.remove();
            addMessageToUI(msg.role, msg.content);
            break;
        case 'startStream':
            isStreaming = true;
            activeContent = ""; 
            activeMessageDiv = document.createElement('div');
            activeMessageDiv.className = `message ${msg.role}`;
            chatContainer.appendChild(activeMessageDiv);
            break;
        case 'streamToken':
            if (activeMessageDiv) {
                activeContent += msg.content;
                updateActiveMessage(activeContent);
            }
            break;
        case 'endStream':
            isStreaming = false;
            activeMessageDiv = null;
            activeContent = "";
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
    messageInput.style.height = 'auto';
}

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
    renderContent(div, content);
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function updateActiveMessage(content) {
    if (!activeMessageDiv) return;
    activeMessageDiv.innerHTML = '';
    renderContent(activeMessageDiv, content);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function renderContent(container, content) {
    // Split by code blocks
    const parts = content.split(/```(\w*)\n([\s\S]*?)```/g);
    
    parts.forEach((part, index) => {
        if (index % 3 === 0) {
            if (part) { // Allow whitespace for streaming spacing
                const wrapper = document.createElement('div');
                wrapper.className = 'markdown-body';
                wrapper.innerHTML = formatMarkdown(part);
                container.appendChild(wrapper);
            }
        } else if (index % 3 === 2) {
            const lang = parts[index-1] || 'text';
            container.appendChild(createCodeBlock(part, lang));
        }
    });

    // Fallback
    if (parts.length === 1) {
        container.innerHTML = `<div class="markdown-body">${formatMarkdown(content)}</div>`;
    }
}

// --- Markdown Parser ---
function formatMarkdown(text) {
    let html = text
        // Escape HTML
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        
        // Headers (### Title)
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        
        // Bold / Italic
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        
        // Inline Code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        
        // Lists (- item)
        .replace(/^\s*[-*]\s+(.*$)/gm, '<li>$1</li>')
        // Numbered Lists (1. item)
        .replace(/^\s*\d+\.\s+(.*$)/gm, '<li>$1</li>')
        
        // Horizontal Rule
        .replace(/^---$/gm, '<hr>');

    // Wrap lists in <ul> (Simple Heuristic: consecutive <li>s)
    // This regex finds a group of <li>...</li> and wraps them.
    // Note: This is a "light" parser, ideal for chat streams.
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    
    // Paragraphs: Double newlines become breaks, but ignore if inside tags
    // We use CSS white-space: pre-wrap usually, but for mixed HTML we need <p> or <br>
    html = html.replace(/\n\n/g, '<br><br>');

    return html;
}

function createCodeBlock(code, lang) {
    const container = document.createElement('div');
    container.className = 'code-block-container';

    const header = document.createElement('div');
    header.className = 'code-header';
    header.innerHTML = `<span class="lang-label">${lang}</span>`;

    const actions = document.createElement('div');
    actions.className = 'code-actions';

    const copyBtn = createBtn('Copy', ICONS.copy, () => {
        navigator.clipboard.writeText(code);
        copyBtn.innerHTML = `${ICONS.check} <span>Copied</span>`;
        setTimeout(() => copyBtn.innerHTML = `${ICONS.copy} <span>Copy</span>`, 2000);
    });

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
    btn.innerHTML = `${icon} <span>${label}</span>`;
    btn.addEventListener('click', onClick);
    return btn;
}