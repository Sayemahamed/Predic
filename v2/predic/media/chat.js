const vscode = acquireVsCodeApi();

// Elements
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const attachButton = document.getElementById('attach-button');
const contextBar = document.getElementById('context-bar');
const modelSelector = document.getElementById('model-selector');

// --- ICONS LIBRARY ---
const ICONS = {
    user: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
    bot: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Z"/><path d="m8 6 8 8"/><path d="m16 6-8 8"/><rect x="3" y="14" width="18" height="8" rx="2"/></svg>`,
    copy: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
    check: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
    insert: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>`,
    close: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
    defaultFile: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>`
};

// Mapping for online icons
const ICON_MAP = {
    'js': 'javascript', 'ts': 'typescript', 'py': 'python',
    'jsx': 'react', 'tsx': 'react_ts', 'html': 'html', 'css': 'css',
    'json': 'json', 'md': 'markdown', 'c': 'c', 'cpp': 'cpp',
    'cs': 'csharp', 'go': 'go', 'rs': 'rust', 'java': 'java',
    'php': 'php', 'rb': 'ruby', 'sql': 'database', 'xml': 'xml',
    'yaml': 'yaml', 'yml': 'yaml', 'sh': 'console', 'bat': 'console',
    'dockerfile': 'docker', 'git': 'git', 'lua': 'lua', 'swift': 'swift',
    'kt': 'kotlin', 'dart': 'dart'
};

// State
let isStreaming = false;
let activeMessageContent = null;

window.addEventListener('message', event => {
    const msg = event.data;
    switch (msg.type) {
        case 'updateModels':
            updateModelDropdown(msg.models);
            break;
        case 'addMessage':
            const welcome = document.querySelector('.welcome-message');
            if (welcome) welcome.remove();
            addMessageToUI(msg.role, msg.content);
            break;
        case 'startStream':
            isStreaming = true;
            createMessageWrapper(msg.role);
            break;
        case 'streamToken':
            if (activeMessageContent) {
                activeMessageContent.dataset.raw = (activeMessageContent.dataset.raw || "") + msg.content;
                renderContent(activeMessageContent, activeMessageContent.dataset.raw);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            break;
        case 'endStream':
            isStreaming = false;
            activeMessageContent = null;
            break;
        case 'fileAttached':
            addContextChip(msg.fileName);
            break;
        case 'clearContext':
            clearContextUI();
            break;
    }
});

function updateModelDropdown(models) {
    modelSelector.innerHTML = '';
    if(models.length === 0) {
        const opt = document.createElement('option');
        opt.text = "No models";
        modelSelector.add(opt);
        return;
    }
    models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.path;
        opt.text = m.name.replace('.gguf', '');
        opt.selected = m.isActive;
        modelSelector.add(opt);
    });
}

modelSelector.addEventListener('change', (e) => {
    vscode.postMessage({ type: 'changeModel', value: e.target.value });
});

function addContextChip(name) {
    contextBar.classList.remove('hidden');
    const ext = name.split('.').pop().toLowerCase();
    
    const iconName = ICON_MAP[ext] || 'file'; 
    const iconUrl = `https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/main/icons/${iconName}.svg`;

    const chip = document.createElement('div');
    chip.className = 'context-chip';
    chip.innerHTML = `
        <img src="${iconUrl}" class="chip-img" onerror="this.style.display='none'; this.nextElementSibling.style.display='block'">
        <span class="chip-fallback" style="display:none">${ICONS.defaultFile}</span>
        <span class="chip-text">${name}</span> 
        <span class="chip-close">${ICONS.close}</span>
    `;
    
    chip.querySelector('.chip-close').onclick = (e) => {
        e.stopPropagation();
        chip.remove();
        if(contextBar.children.length === 0) {
            contextBar.classList.add('hidden');
            vscode.postMessage({ type: 'removeContext' });
        }
    };
    contextBar.appendChild(chip);
}

function clearContextUI() {
    contextBar.innerHTML = '';
    contextBar.classList.add('hidden');
}

function createMessageWrapper(role) {
    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${role}`;
    
    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = `${ICONS[role === 'user' ? 'user' : 'bot']} <span>${role === 'user' ? 'You' : 'Predic'}</span>`;
    
    activeMessageContent = document.createElement('div');
    activeMessageContent.className = 'message-content';
    
    wrapper.append(header, activeMessageContent);
    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addMessageToUI(role, content) {
    createMessageWrapper(role);
    renderContent(activeMessageContent, content);
    activeMessageContent = null;
}

function renderContent(container, content) {
    container.innerHTML = '';
    const parts = content.split(/```(\w*)\n([\s\S]*?)```/g);
    parts.forEach((part, index) => {
        if (index % 3 === 0) {
            if (part) {
                const p = document.createElement('div');
                p.className = 'markdown-body';
                p.innerHTML = formatMarkdown(part);
                container.appendChild(p);
            }
        } else if (index % 3 === 2) {
            const lang = parts[index-1] || 'text';
            container.appendChild(createCodeBlock(part, lang));
        }
    });
    if (parts.length === 1) {
        container.innerHTML = `<div class="markdown-body">${formatMarkdown(content)}</div>`;
    }
}

function formatMarkdown(text) {
    return text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

function createCodeBlock(code, lang) {
    const container = document.createElement('div');
    container.className = 'code-block-container';
    const header = document.createElement('div');
    header.className = 'code-header';
    header.innerHTML = `<span class="lang-label">${lang}</span>`;
    const actions = document.createElement('div');
    actions.className = 'code-actions';
    const copyBtn = createBtn('Copy', ICONS.copy, () => navigator.clipboard.writeText(code));
    const insertBtn = createBtn('Insert', ICONS.insert, () => vscode.postMessage({ type: 'insertCode', value: code }));
    actions.append(copyBtn, insertBtn);
    header.append(actions);
    const pre = document.createElement('pre');
    pre.textContent = code;
    container.append(header, pre);
    return container;
}

function createBtn(title, icon, onClick) {
    const btn = document.createElement('button');
    btn.className = 'action-btn';
    btn.title = title;
    btn.innerHTML = icon;
    btn.addEventListener('click', onClick);
    return btn;
}

function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;
    vscode.postMessage({ type: 'sendMessage', value: text });
    messageInput.value = '';
    messageInput.style.height = 'auto';
}

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

sendButton.addEventListener('click', sendMessage);
attachButton.addEventListener('click', () => vscode.postMessage({ type: 'attachFile' }));

vscode.postMessage({ type: 'refreshModels' });