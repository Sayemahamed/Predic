import * as vscode from 'vscode';
import { PredicAPIClient } from '../api/client';
import { ChatMessage, Model } from '../types';

export class ChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'predic.chat';
    private _view?: vscode.WebviewView;
    private apiClient: PredicAPIClient;
    private messages: ChatMessage[] = [];
    private availableModels: Model[] = [];

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private context: vscode.ExtensionContext
    ) {
        this.apiClient = new PredicAPIClient();
        this.loadModels();
    }

    private async loadModels() {
        try {
            this.availableModels = await this.apiClient.getAvailableModels();
            this.updateWebview();
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async data => {
            switch (data.type) {
                case 'sendMessage':
                    await this.handleSendMessage(data.message);
                    break;
                case 'clear':
                    this.messages = [];
                    this.updateWebview();
                    break;
                case 'selectModel':
                    await this.handleSelectModel(data.modelId);
                    break;
                case 'refreshModels':
                    await this.loadModels();
                    break;
            }
        });

        // Initial update
        this.updateWebview();
    }

    private async handleSelectModel(modelId: string) {
        const config = vscode.workspace.getConfiguration('predic');
        await config.update('selectedModel', modelId, vscode.ConfigurationTarget.Global);
        vscode.window.showInformationMessage(`Selected model: ${modelId}`);
        this.updateWebview();
    }

    private async handleSendMessage(message: string) {
        const config = vscode.workspace.getConfiguration('predic');
        const selectedModel = config.get('selectedModel', '');
        
        if (!selectedModel) {
            vscode.window.showErrorMessage('Please select a model first');
            return;
        }

        // Add user message
        const userMessage: ChatMessage = {
            role: 'user',
            content: message,
            timestamp: Date.now()
        };
        this.messages.push(userMessage);
        this.updateWebview();

        try {
            // Send to API
            const response = await this.apiClient.chat({
                model_id: selectedModel,
                messages: this.messages
            });

            // Add assistant response
            this.messages.push({
                ...response,
                timestamp: Date.now()
            });
            this.updateWebview();
        } catch (error) {
            vscode.window.showErrorMessage('Failed to send message');
        }
    }

    private updateWebview() {
        if (this._view) {
            const config = vscode.workspace.getConfiguration('predic');
            const selectedModel = config.get('selectedModel', '');
            
            this._view.webview.postMessage({
                type: 'update',
                messages: this.messages,
                models: this.availableModels.filter(m => m.status === 'ready'),
                selectedModel: selectedModel
            });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.css')
        );
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.js')
        );
        const codiconsUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'node_modules', '@vscode/codicons', 'dist', 'codicon.css')
        );

        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${codiconsUri}" rel="stylesheet">
            <link href="${styleUri}" rel="stylesheet">
            <title>Predic Chat</title>
        </head>
        <body>
            <div class="chat-container">
                <div class="chat-header">
                    <div class="header-left">
                        <i class="codicon codicon-hubot"></i>
                        <h3>Predic AI Assistant</h3>
                    </div>
                    <div class="header-right">
                        <select id="modelSelector" class="model-selector">
                            <option value="">Select a model...</option>
                        </select>
                        <button id="refreshModelsBtn" class="icon-button" title="Refresh models">
                            <i class="codicon codicon-refresh"></i>
                        </button>
                    </div>
                </div>
                
                <div class="messages" id="messages">
                    <div class="welcome-message">
                        <i class="codicon codicon-hubot large-icon"></i>
                        <h2>Welcome to Predic!</h2>
                        <p>Select a model above to start chatting with your AI coding assistant.</p>
                    </div>
                </div>
                
                <div class="input-container">
                    <div class="input-wrapper">
                        <textarea 
                            id="messageInput" 
                            placeholder="Ask me about your code..."
                            rows="3"
                        ></textarea>
                        <div class="input-actions">
                            <button id="sendButton" class="primary-button" disabled>
                                <i class="codicon codicon-send"></i>
                                Send
                            </button>
                            <button id="clearButton" class="secondary-button">
                                <i class="codicon codicon-clear-all"></i>
                                Clear
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}