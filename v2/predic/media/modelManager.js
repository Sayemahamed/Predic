(function() {
    const vscode = acquireVsCodeApi();
    const modelList = document.getElementById('modelList');
    const refreshButton = document.getElementById('refreshButton');
    
    let models = [];
    let selectedModel = '';

    refreshButton.addEventListener('click', () => {
        vscode.postMessage({ type: 'refreshModels' });
    });

    function renderModels() {
        modelList.innerHTML = '';
        
        models.forEach(model => {
            const modelCard = document.createElement('div');
            modelCard.className = 'model-card';
            if (model.id === selectedModel) {
                modelCard.classList.add('selected');
            }
            
            const modelInfo = document.createElement('div');
            modelInfo.className = 'model-info';
            modelInfo.innerHTML = `
                <h3>${model.name}</h3>
                <p>Size: ${model.size}</p>
                <p>Status: <span class="status-badge ${model.status}">${model.status}</span></p>
            `;
            
            const modelActions = document.createElement('div');
            modelActions.className = 'model-actions';
            
            if (model.status === 'downloading') {
                const progressBar = document.createElement('div');
                progressBar.className = 'progress-bar';
                const progressFill = document.createElement('div');
                progressFill.className = 'progress-fill';
                progressFill.style.width = `${model.progress}%`;
                progressBar.appendChild(progressFill);
                modelActions.appendChild(progressBar);
            } else if (model.status === 'available') {
                const downloadButton = document.createElement('button');
                downloadButton.textContent = 'Download';
                downloadButton.onclick = () => downloadModel(model.id);
                modelActions.appendChild(downloadButton);
            } else if (model.status === 'ready') {
                const selectButton = document.createElement('button');
                selectButton.textContent = model.id === selectedModel ? 'Selected' : 'Select';
                selectButton.disabled = model.id === selectedModel;
                selectButton.onclick = () => selectModel(model.id);
                modelActions.appendChild(selectButton);
            }
            
            modelCard.appendChild(modelInfo);
            modelCard.appendChild(modelActions);
            modelList.appendChild(modelCard);
        });
    }

    function downloadModel(modelId) {
        vscode.postMessage({
            type: 'downloadModel',
            modelId: modelId
        });
    }

    function selectModel(modelId) {
        vscode.postMessage({
            type: 'selectModel',
            modelId: modelId
        });
    }

    window.addEventListener('message', event => {
        const message = event.data;
        switch (message.type) {
            case 'updateModels':
                models = message.models;
                selectedModel = message.selectedModel;
                renderModels();
                break;
            case 'updateModelStatus':
                const modelIndex = models.findIndex(m => m.id === message.modelId);
                if (modelIndex !== -1) {
                    models[modelIndex] = { ...models[modelIndex], ...message.status };
                    renderModels();
                }
                break;
            case 'modelSelected':
                selectedModel = message.modelId;
                renderModels();
                break;
        }
    });
})();