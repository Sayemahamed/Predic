const vscode = acquireVsCodeApi();

// DOM Elements
const elKoboldPath = document.getElementById('koboldPath');
const elModelDir = document.getElementById('modelDir');
const elModelList = document.getElementById('model-list');

// Buttons
document.getElementById('btn-kobold-path').addEventListener('click', () => {
    vscode.postMessage({ type: 'selectPath', target: 'koboldCppPath' });
});

document.getElementById('btn-model-dir').addEventListener('click', () => {
    vscode.postMessage({ type: 'selectPath', target: 'modelDir' });
});

// Handle Messages from Extension
window.addEventListener('message', event => {
    const message = event.data;
    if (message.type === 'updateState') {
        renderState(message.data);
    }
});

function renderState(data) {
    elKoboldPath.value = data.koboldPath || "Auto-detected / Default";
    elModelDir.value = data.modelDir || "Auto-detected / Default";
    
    elModelList.innerHTML = '';
    
    if (data.models.length === 0) {
        elModelList.innerHTML = `<div class="empty-state">No .gguf models found in the directory.</div>`;
        return;
    }

    data.models.forEach(model => {
        const card = document.createElement('div');
        card.className = `model-item ${model.isActive ? 'active' : ''}`;
        
        card.innerHTML = `
            <div class="model-info">
                <span class="model-icon">📦</span>
                <span class="model-name">${model.name}</span>
                ${model.isActive ? '<span class="badge">Active</span>' : ''}
            </div>
            <div class="model-actions">
                ${!model.isActive ? `<button class="btn-load">Load</button>` : '<button disabled>Running</button>'}
            </div>
        `;

        if (!model.isActive) {
            card.querySelector('.btn-load').addEventListener('click', () => {
                vscode.postMessage({ type: 'setActiveModel', path: model.path });
            });
        }

        elModelList.appendChild(card);
    });
}

// Initial Request
vscode.postMessage({ type: 'refresh' });