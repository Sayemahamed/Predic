const vscode = acquireVsCodeApi();

// Elements
const chatContainer = document.getElementById('chat-container');
const messageInput = document.getElementById('message-input');
const highlightBackdrop = document.getElementById('highlight-backdrop');
const sendButton = document.getElementById('send-button');
const attachButton = document.getElementById('attach-button');
const contextBar = document.getElementById('context-bar');
const modelSelector = document.getElementById('model-selector');

// Header Buttons
const btnFullScreen = document.getElementById('btn-fullscreen');
const btnSettings = document.getElementById('btn-settings');
const btnClose = document.getElementById('btn-close');
const btnOpen = document.getElementById('btn-open');

// Header Listeners
if (btnFullScreen) btnFullScreen.addEventListener('click', () => vscode.postMessage({ type: 'toggleFullScreen' }));
if (btnSettings) btnSettings.addEventListener('click', () => vscode.postMessage({ type: 'openSettings' }));
if (btnClose) btnClose.addEventListener('click', () => vscode.postMessage({ type: 'closeChat' }));
if (btnOpen) btnOpen.addEventListener('click', () => vscode.postMessage({ type: 'openInEditor' }));

// Icons
const ICONS = {
    user: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
    bot: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Z"/><path d="m8 6 8 8"/><path d="m16 6-8 8"/><rect x="3" y="14" width="18" height="8" rx="2"/></svg>`,
    brain: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/></svg>`,
    chevron: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>`,
    copy: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
    edit: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>`,
    check: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
    insert: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>`,
    close: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
    defaultFile: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>`
};

// Icon Mapping
const ICON_MAP = {
    'js': 'javascript', 'ts': 'typescript', 'py': 'python', 'jsx': 'react', 'tsx': 'react_ts', 
    'html': 'html', 'css': 'css', 'json': 'json', 'md': 'markdown', 'c': 'c', 'cpp': 'cpp',
    'cs': 'csharp', 'go': 'go', 'rs': 'rust', 'java': 'java', 'php': 'php', 'rb': 'ruby', 
    'sql': 'database', 'xml': 'xml', 'yaml': 'yaml', 'yml': 'yaml', 'sh': 'console', 
    'dockerfile': 'docker', 'git': 'git', 'lua': 'lua', 'swift': 'swift', 'kt': 'kotlin', 'dart': 'dart'
};

// State
let isStreaming = false;
let activeMessageContent = null;
let currentLanguageHint = 'text';

window.addEventListener('message', event => {
    const msg = event.data;
    switch (msg.type) {
        case 'updateModels':
            updateModelDropdown(msg.models);
            break;
        case 'addMessage':
            document.querySelector('.welcome-message')?.remove();
            addMessageToUI(msg.role, msg.content);
            break;
        case 'startStream':
            isStreaming = true;
            currentLanguageHint = msg.language || 'text';
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
    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${role}`;
    
    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = `${ICONS[role === 'user' ? 'user' : 'bot']} <span>${role === 'user' ? 'You' : 'Predic'}</span>`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (role === 'user') {
        const displayContent = content.replace(/@([a-zA-Z0-9_\-\.]+)/g, '<span class="file-ref">@$1</span>');
        contentDiv.innerHTML = displayContent;

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'user-actions';
        
        const btnEdit = document.createElement('button');
        btnEdit.className = 'msg-action-btn';
        btnEdit.innerHTML = `${ICONS.edit} Edit`;
        btnEdit.onclick = () => {
            messageInput.value = content;
            messageInput.focus();
            syncBackdrop();
        };

        const btnCopy = document.createElement('button');
        btnCopy.className = 'msg-action-btn';
        btnCopy.innerHTML = `${ICONS.copy} Copy`;
        btnCopy.onclick = () => navigator.clipboard.writeText(content);

        actionsDiv.append(btnEdit, btnCopy);
        wrapper.append(header, contentDiv, actionsDiv);
    } else {
        renderContent(contentDiv, content);
        wrapper.append(header, contentDiv);
    }

    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// --- CORE RENDERING LOGIC ---
function renderContent(container, content) {
    container.innerHTML = '';
    
    // 1. EXTRACT & RENDER "THINKING" BLOCK (<think>...</think>)
    const thinkMatch = content.match(/<think>([\s\S]*?)(?:<\/think>|$)/);
    let mainContent = content;

    if (thinkMatch) {
        const thinkText = thinkMatch[1].trim();
        if (thinkText) {
            const thinkDetails = document.createElement('details');
            thinkDetails.className = 'thinking-block';
            
            // Auto-open only while streaming the thinking part
            if (isStreaming && !content.includes('</think>')) {
                 thinkDetails.open = true;
            }

            const summary = document.createElement('summary');
            summary.innerHTML = `${ICONS.brain} Thinking Process ${ICONS.chevron}`;
            
            const p = document.createElement('div');
            p.className = 'thinking-content';
            p.innerText = thinkText; 
            
            thinkDetails.appendChild(summary);
            thinkDetails.appendChild(p);
            container.appendChild(thinkDetails);
        }
        // Remove the think block from the main content so we don't render it twice
        mainContent = content.replace(thinkMatch[0], '');
    }

    // 2. STRIP DUPLICATE NAMES (Fixes "Predic is showing twice")
    // If the model output starts with "Predic:", "Assistant:", or "Bot:", remove it.
    mainContent = mainContent.replace(/^\s*(Predic|Assistant|Bot|AI):\s*/i, '');

    // 3. DYNAMIC AUTO-FIX FOR CODE BLOCKS (Language Detection)
    let contentToProcess = mainContent;
    const hasCodeBlock = /```/.test(mainContent);
    
    if (!hasCodeBlock && mainContent.trim().length > 0) {
        contentToProcess = `\`\`\`${currentLanguageHint}\n${mainContent}\n\`\`\``;
    }

    // 4. MARKDOWN & CODE BLOCK RENDERING
    // Regex to split by code blocks (Handles Windows \r\n and Standard \n)
    const regex = /```(\w*)\r?\n([\s\S]*?)```/g;
    
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(contentToProcess)) !== null) {
        const preText = contentToProcess.substring(lastIndex, match.index);
        if (preText.trim()) {
            const p = document.createElement('div');
            p.className = 'markdown-body';
            p.innerHTML = formatMarkdown(preText);
            container.appendChild(p);
        }

        const lang = match[1] || 'text';
        const code = match[2];
        container.appendChild(createCodeBlock(code, lang));

        lastIndex = regex.lastIndex;
    }

    const remaining = contentToProcess.substring(lastIndex);
    if (remaining.trim()) {
        // Handle unclosed blocks during streaming
        const openBlockMatch = remaining.match(/```(\w*)\r?\n([\s\S]*)$/);
        
        if (openBlockMatch) {
            const preText = remaining.substring(0, openBlockMatch.index);
            if (preText.trim()) {
                const p = document.createElement('div');
                p.className = 'markdown-body';
                p.innerHTML = formatMarkdown(preText);
                container.appendChild(p);
            }

            const lang = openBlockMatch[1] || 'text';
            const code = openBlockMatch[2];
            container.appendChild(createCodeBlock(code, lang));
        } else {
            const p = document.createElement('div');
            p.className = 'markdown-body';
            p.innerHTML = formatMarkdown(remaining);
            container.appendChild(p);
        }
    }
}

// IMPROVED MARKDOWN FORMATTER
function formatMarkdown(text) {
    let html = text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/__(.*?)__/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/^\s*[-*]\s+(.*$)/gm, '<li>$1</li>')
        .replace(/^\s*\d+\.\s+(.*$)/gm, '<li>$1</li>');

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

function syncBackdrop() {
    const text = messageInput.value;
    const highlighted = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/@([a-zA-Z0-9_\-\.]+)/g, '<span class="file-ref">@$1</span>');

    highlightBackdrop.innerHTML = highlighted + (text.endsWith('\n') ? '<br>&nbsp;' : '');
    
    messageInput.style.height = 'auto';
    const h = messageInput.scrollHeight;
    messageInput.style.height = h + 'px';
    highlightBackdrop.style.height = h + 'px';
}

function syncScroll() {
    highlightBackdrop.scrollTop = messageInput.scrollTop;
}

messageInput.addEventListener('input', syncBackdrop);
messageInput.addEventListener('scroll', syncScroll);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;
    vscode.postMessage({ type: 'sendMessage', value: text });
    messageInput.value = '';
    syncBackdrop();
}

sendButton.addEventListener('click', sendMessage);
attachButton.addEventListener('click', () => vscode.postMessage({ type: 'attachFile' }));

vscode.postMessage({ type: 'refreshModels' });