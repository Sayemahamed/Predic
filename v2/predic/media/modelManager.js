const vscode = acquireVsCodeApi();

const elKoboldPath = document.getElementById('koboldPath');
const elModelDir = document.getElementById('modelDir');
const elLocalList = document.getElementById('local-list');
const elCuratedList = document.getElementById('curated-list');
const elPredicList = document.getElementById('predic-list');

// Event Listeners
document.getElementById('btn-kobold-path').addEventListener('click', () => {
    vscode.postMessage({ type: 'selectPath', target: 'koboldCppPath' });
});

document.getElementById('btn-model-dir').addEventListener('click', () => {
    vscode.postMessage({ type: 'selectPath', target: 'modelDir' });
});

window.addEventListener('message', event => {
    const message = event.data;
    if (message.type === 'updateState') {
        renderState(message.data);
    }
});

function renderState(data) {
    // Settings
    elKoboldPath.value = data.koboldPath || "";
    elModelDir.value = data.modelDir || "";
    
    // Helper for rendering cards
    const renderCards = (models, container, isPredicList) => {
        container.innerHTML = '';
        models.forEach(model => {
            const card = document.createElement('div');
            card.className = `model-card ${model.isActive ? 'active-card' : ''} ${isPredicList ? 'predic-card' : ''}`;
            
            let btnHtml = '';
            if (model.isActive) {
                btnHtml = `<button class="btn-active" disabled>Active</button>`;
            } else if (model.isDownloaded) {
                btnHtml = `<button class="btn-load" data-path="${model.filename}">Load</button>`;
            } else {
                btnHtml = `<button class="btn-download" data-id="${model.id}">Download</button>`;
            }

            card.innerHTML = `
                <div class="card-header">
                    <span class="model-title">${model.name}</span>
                    <span class="model-size">${model.size}</span>
                </div>
                <p class="model-desc">${model.description}</p>
                <div class="card-footer">
                    ${btnHtml}
                </div>
            `;
            
            const btn = card.querySelector('button');
            if (btn.classList.contains('btn-download')) {
                btn.addEventListener('click', () => {
                    btn.innerText = "Downloading...";
                    btn.disabled = true;
                    // Pass 'isPredic' flag so backend knows which list to look in
                    vscode.postMessage({ type: 'downloadModel', modelId: model.id, isPredic: isPredicList });
                });
            } else if (btn.classList.contains('btn-load')) {
                btn.addEventListener('click', () => {
                    const matchedLocal = data.localModels.find(l => l.name === model.filename);
                    if (matchedLocal) {
                        vscode.postMessage({ type: 'setActiveModel', path: matchedLocal.path });
                    } else {
                        // Should not happen if logic is correct
                        console.error("File marked downloaded but not found locally.");
                    }
                });
            }

            container.appendChild(card);
        });
    };

    // Render Predic Models
    renderCards(data.predicModels, elPredicList, true);

    // Render Curated Models
    renderCards(data.curatedModels, elCuratedList, false);

    // Render Local List
    elLocalList.innerHTML = '';
    if (data.localModels.length === 0) {
        elLocalList.innerHTML = `<div class="empty-state">No models found in directory.</div>`;
    } else {
        data.localModels.forEach(model => {
            const row = document.createElement('div');
            row.className = `model-row ${model.isActive ? 'active' : ''}`;
            
            row.innerHTML = `
                <div class="row-info">
                    <span class="icon-box">📦</span>
                    <span class="row-name">${model.name}</span>
                    ${model.isActive ? '<span class="badge">Running</span>' : ''}
                </div>
                ${!model.isActive ? `<button class="btn-secondary-sm">Load</button>` : ''}
            `;

            if (!model.isActive) {
                row.querySelector('button').addEventListener('click', () => {
                    vscode.postMessage({ type: 'setActiveModel', path: model.path });
                });
            }
            elLocalList.appendChild(row);
        });
    }
}

// Init
vscode.postMessage({ type: 'refresh' });