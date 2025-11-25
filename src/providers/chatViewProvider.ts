import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { PredicAPIClient } from '../api/client';

export class ChatViewProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private apiClient: PredicAPIClient;
    private activeContext: string = "";
    private activeFileName: string = "";

    constructor(private readonly _extensionUri: vscode.Uri) {
        this.apiClient = new PredicAPIClient();
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

        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'sendMessage':
                    await this.handleUserMessage(data.value);
                    break;
                case 'attachFile':
                    await this.handleFileAttachment();
                    break;
                case 'removeContext':
                    this.activeContext = "";
                    this.activeFileName = "";
                    break;
                case 'insertCode':
                    await this.handleInsertCode(data.value);
                    break;
                case 'changeModel':
                    await this.setActiveModel(data.value);
                    break;
                case 'refreshModels':
                    this.sendModelList();
                    break;
                // --- HEADER ACTIONS ---
                case 'openSettings':
                    vscode.commands.executeCommand('predic.openModelManager');
                    break;
                case 'toggleFullScreen':
                    vscode.commands.executeCommand('workbench.action.toggleFullScreen');
                    break;
                case 'closeChat':
                    vscode.commands.executeCommand('workbench.action.closeSidebar');
                    break;
                case 'openInEditor':
                    vscode.commands.executeCommand('workbench.action.moveView');
                    break;
                case 'clearChat':
                    this.activeContext = "";
                    break;
            }
        });

        this.sendModelList();
    }

    public clearChat() {
        if (this._view) {
            this._view.webview.postMessage({ type: 'clearChat' });
            this.activeContext = "";
        }
    }

    public openInEditor() {
        vscode.commands.executeCommand('workbench.action.moveView');
    }

    private sendModelList() {
        if (!this._view) return;
        const config = vscode.workspace.getConfiguration('predic');
        const modelDir = config.get<string>('modelDir') || path.join(this._extensionUri.fsPath, '..', 'models');
        const currentPath = config.get<string>('modelPath') || "";

        let models: any[] = [];
        if (fs.existsSync(modelDir)) {
            try {
                models = fs.readdirSync(modelDir)
                    .filter(f => f.endsWith('.gguf'))
                    .map(f => ({
                        name: f,
                        path: path.join(modelDir, f),
                        isActive: path.normalize(path.join(modelDir, f)) === path.normalize(currentPath)
                    }));
            } catch (e) {
                console.error("Error reading models dir:", e);
            }
        }
        
        this._view.webview.postMessage({ type: 'updateModels', models: models });
    }

    private async setActiveModel(modelPath: string) {
        await vscode.workspace.getConfiguration('predic').update('modelPath', modelPath, vscode.ConfigurationTarget.Global);
        vscode.commands.executeCommand('predic.restartServer');
        vscode.window.showInformationMessage(`Switched model. Restarting server...`);
        this.sendModelList();
    }

    public async triggerAnalysis(prompt: string, codeContext: string) {
        if (this._view) {
            this._view.show?.(true);
            this.activeContext = codeContext;
            this.activeFileName = "Selection";
            
            this._view.webview.postMessage({ 
                type: 'fileAttached', 
                fileName: "Selection" 
            });

            await this.handleUserMessage(prompt);
        }
    }

    private async handleFileAttachment() {
        const uris = await vscode.window.showOpenDialog({
            canSelectMany: false,
            openLabel: 'Add Context',
            filters: { 
                'Code Files': [
                    'ts', 'js', 'jsx', 'tsx', 'py', 'java', 'c', 'cpp', 'h', 'hpp', 'cs', 'go', 'rs', 
                    'php', 'rb', 'sh', 'yaml', 'yml', 'json', 'md', 'txt', 'css', 'html', 'xml', 'sql', 
                    'lua', 'pl', 'ex', 'hs', 'clj', 'groovy', 'nim', 'cr', 'jl', 'ml', 'fs', 'zig', 
                    'elm', 'erl', 'vhd', 'v', 'f90', 'asm', 'toml', 'swift', 'kt', 'scala', 'dart', 'r'
                ],
                'All Files': ['*']
            }
        });

        if (uris && uris[0]) {
            try {
                const content = fs.readFileSync(uris[0].fsPath, 'utf-8');
                this.activeFileName = uris[0].path.split('/').pop() || "File";
                this.activeContext = content;
                
                this._view?.webview.postMessage({ 
                    type: 'fileAttached', 
                    fileName: this.activeFileName 
                });
            } catch (error: any) {
                vscode.window.showErrorMessage(`Failed to read file: ${error.message}`);
            }
        }
    }

    private async handleInsertCode(code: string) {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            await editor.edit(editBuilder => {
                if (!editor.selection.isEmpty) {
                    editBuilder.replace(editor.selection, code);
                } else {
                    editBuilder.insert(editor.selection.active, code);
                }
            });
        } else {
            vscode.window.showErrorMessage('Open a file to accept this code.');
        }
    }

    private async handleUserMessage(content: string) {
        if (!this._view) return;

        this._view.webview.postMessage({ type: 'addMessage', role: 'user', content: content });
        this._view.webview.postMessage({ type: 'startStream', role: 'assistant' });

        try {
            let fullPrompt = "";
            if (this.activeContext) {
                fullPrompt += `Context (${this.activeFileName}):\n\`\`\`\n${this.activeContext}\n\`\`\`\n`;
            }
            fullPrompt += `\nUser Query: ${content}`;

            const messages = [
                { role: "system", content: "You are Predic. Provide ONLY code or direct answers. Do NOT explain code unless explicitly asked." },
                { role: "user", content: fullPrompt }
            ];
            
            await this.apiClient.chatStream(messages, (token) => {
                this._view?.webview.postMessage({ type: 'streamToken', content: token });
            });
            
            this._view.webview.postMessage({ type: 'endStream' });

            if(this.activeContext) {
                this.activeContext = "";
                this._view.webview.postMessage({ type: 'clearContext' });
            }

        } catch (error: any) {
            this._view.webview.postMessage({ type: 'endStream' });
            this._view.webview.postMessage({ 
                type: 'addMessage', 
                role: 'system', 
                content: `Error: ${error.message || "Connection failed."}` 
            });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'scripts', 'chat.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'styles', 'chat.css'));
        const logoUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'assets', 'background.png'));
        const nonce = getNonce();

        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} https: data:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>Predic Chat</title>
        </head>
        <body>
            <div id="chat-container">
                <div class="welcome-message">
                    <img src="${logoUri}" class="chat-logo" alt="Logo">
                    <h2>Predic AI</h2>
                    <p>Your local coding assistant.</p>
                </div>
            </div>
            
            <div class="input-wrapper">
                <div id="context-bar" class="hidden"></div>
                
                <div class="input-container">
                    <div class="input-stack">
                        <div id="highlight-backdrop" aria-hidden="true"></div>
                        <textarea id="message-input" placeholder="Ask about your code or @file..."></textarea>
                    </div>

                    <div class="input-footer">
                        <div class="left-tools">
                            <button id="attach-button" class="icon-btn" title="Attach Context">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
                            </button>
                            <select id="model-selector" class="model-dropdown">
                                <option disabled selected>Loading...</option>
                            </select>
                        </div>
                        <div class="right-tools">
                            <button id="send-button" class="send-btn" title="Send">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" x2="11" y1="2" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            <script nonce="${nonce}" src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}

function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}