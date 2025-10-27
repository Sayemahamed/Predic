import * as vscode from 'vscode';
import { PredicAPIClient } from '../api/client';
import { Model } from '../types';

export class ModelManagerProvider {
    private panel: vscode.WebviewPanel | undefined;
    private apiClient: PredicAPIClient;

    constructor(
        private readonly extensionUri: vscode.Uri,
        private context: vscode.ExtensionContext
    ) {
        this.apiClient = new PredicAPIClient();
    }

    public async show() {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        if (this.panel) {
            this.panel.reveal(column);
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'predicModelManager',
            'Predic Model Manager',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [this.extensionUri]
            }
        );

        this.panel.webview.html = await this._getHtmlForWebview(this.panel.webview);

        this.panel.webview.onDidReceiveMessage(
            async message => {
                switch (message.type) {
                    case 'downloadModel':
                        await this.handleDownloadModel(message.modelId);
                        break;
                    case 'selectModel':
                        await this.handleSelectModel(message.modelId);
                        break;
                    case 'refreshModels':
                        await this.refreshModels();
                        break;
                }
            },
            undefined,
            this.context.subscriptions
        );

        this.panel.onDidDispose(
            () => {
                this.panel = undefined;
            },
            undefined,
            this.context.subscriptions
        );

        // Load initial models
        await this.refreshModels();
    }

    private async handleDownloadModel(modelId: string) {
        try {
            await this.apiClient.downloadModel(modelId);
            vscode.window.showInformationMessage(`Started downloading model: ${modelId}`);
            this.startPollingStatus(modelId);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to download model: ${error}`);
        }
    }

    private async handleSelectModel(modelId: string) {
        const config = vscode.workspace.getConfiguration('predic');
        await config.update('selectedModel', modelId, vscode.ConfigurationTarget.Global);
        vscode.window.showInformationMessage(`Selected model: ${modelId}`);
        
        if (this.panel) {
            this.panel.webview.postMessage({
                type: 'modelSelected',
                modelId: modelId
            });
        }
    }

    private async refreshModels() {
        try {
            const models = await this.apiClient.getAvailableModels();
            const config = vscode.workspace.getConfiguration('predic');
            const selectedModel = config.get('selectedModel', '');
            
            if (this.panel) {
                this.panel.webview.postMessage({
                    type: 'updateModels',
                    models: models,
                    selectedModel: selectedModel
                });
            }
        } catch (error) {
            vscode.window.showErrorMessage('Failed to load models');
        }
    }

    private startPollingStatus(modelId: string) {
        const interval = setInterval(async () => {
            try {
                const status = await this.apiClient.getModelStatus(modelId);
                
                if (this.panel) {
                    this.panel.webview.postMessage({
                        type: 'updateModelStatus',
                        modelId: modelId,
                        status: status
                    });
                }

                if (status.status === 'ready' || status.status === 'error') {
                    clearInterval(interval);
                    if (status.status === 'ready') {
                        vscode.window.showInformationMessage(`Model ${modelId} is ready!`);
                    } else {
                        vscode.window.showErrorMessage(`Failed to download model ${modelId}`);
                    }
                    await this.refreshModels();
                }
            } catch (error) {
                clearInterval(interval);
            }
        }, 2000);
    }

    private async _getHtmlForWebview(webview: vscode.Webview): Promise<string> {
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this.extensionUri, 'media', 'modelManager.css')
        );
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this.extensionUri, 'media', 'modelManager.js')
        );

        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>Model Manager</title>
        </head>
        <body>
            <div class="container">
                <h1>Predic Model Manager</h1>
                <div class="toolbar">
                    <button id="refreshButton">Refresh</button>
                </div>
                <div id="modelList" class="model-list"></div>
            </div>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}