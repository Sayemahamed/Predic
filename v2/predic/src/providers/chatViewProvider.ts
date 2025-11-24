import * as vscode from 'vscode';
import * as fs from 'fs';
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
                case 'insertCode':
                    await this.handleInsertCode(data.value);
                    break;
            }
        });
    }

    public async triggerAnalysis(prompt: string, codeContext: string) {
        if (this._view) {
            this._view.show?.(true);
            this.activeContext = codeContext;
            this.activeFileName = "Selected Code";
            
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
            filters: { 'Code': ['ts', 'js', 'py', 'java', 'c', 'cpp', 'json', 'md', 'txt', 'html', 'css', 'rs', 'go'] }
        });

        if (uris && uris[0]) {
            this.loadFileContext(uris[0]);
        }
    }

    private async loadFileContext(uri: vscode.Uri) {
        try {
            const content = fs.readFileSync(uri.fsPath, 'utf-8');
            this.activeFileName = uri.path.split('/').pop() || "File";
            this.activeContext = content;
            
            this._view?.webview.postMessage({ 
                type: 'fileAttached', 
                fileName: this.activeFileName 
            });
        } catch (error: any) {
            vscode.window.showErrorMessage(`Failed to read file: ${error.message}`);
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

    private async parseAndLoadFileRefs(content: string): Promise<string> {
        const fileRegex = /@([a-zA-Z0-9_\-\.]+)/g;
        const matches = [...content.matchAll(fileRegex)];
        let loadedContext = "";

        for (const match of matches) {
            const fileName = match[1];
            const files = await vscode.workspace.findFiles(`**/${fileName}`, '**/node_modules/**', 1);
            if (files.length > 0) {
                try {
                    const fileContent = fs.readFileSync(files[0].fsPath, 'utf-8');
                    loadedContext += `\nContent of ${fileName}:\n\`\`\`\n${fileContent}\n\`\`\`\n`;
                    this._view?.webview.postMessage({ type: 'fileAttached', fileName: fileName });
                } catch (e) {
                    console.error(`Could not read ${fileName}`);
                }
            }
        }
        return loadedContext;
    }

    private async handleUserMessage(content: string) {
        if (!this._view) return;

        this._view.webview.postMessage({ type: 'addMessage', role: 'user', content: content });
        this._view.webview.postMessage({ type: 'setLoading', value: true });

        try {
            const dynamicContext = await this.parseAndLoadFileRefs(content);
            let fullPrompt = "";
            
            if (this.activeContext) {
                fullPrompt += `Context (${this.activeFileName}):\n\`\`\`\n${this.activeContext}\n\`\`\`\n`;
            }
            
            if (dynamicContext) {
                fullPrompt += `\nReferenced Files:\n${dynamicContext}\n`;
            }

            fullPrompt += `\nUser Query: ${content}`;

            const messages = [
                { 
                    role: "system", 
                    content: "You are Predic, an intelligent coding companion. When providing code, use Markdown code blocks with the language specified. Do NOT provide explanations unless the user specifically asks 'explain' or 'why'. If you are fixing code, just provide the fixed code block. Be concise." 
                },
                { role: "user", content: fullPrompt }
            ];
            
            const response = await this.apiClient.chat(messages);
            const aiText = response.data.choices[0].message.content;
            
            this._view.webview.postMessage({ type: 'addMessage', role: 'assistant', content: aiText });
            
            if(this.activeContext) {
                this.activeContext = "";
                this._view.webview.postMessage({ type: 'clearContext' });
            }

        } catch (error: any) {
            this._view.webview.postMessage({ 
                type: 'addMessage', 
                role: 'system', 
                content: `Error: ${error.message || "Connection failed. Is the server running?"}` 
            });
        } finally {
            this._view.webview.postMessage({ type: 'setLoading', value: false });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.css'));
        
        // --- NEW: Get Logo URI ---
        const logoPath = vscode.Uri.joinPath(this._extensionUri, 'media', 'logo.png');
        const logoUri = webview.asWebviewUri(logoPath);
        
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
                    <img src="${logoUri}" alt="Predic Logo" class="chat-logo" />
                    <h2>Predic AI</h2>
                    <p>Ask me anything about your code or type <code>@filename</code> to add context.</p>
                </div>
            </div>
            
            <div id="typing-indicator" class="hidden">
                <span></span><span></span><span></span>
            </div>

            <div class="input-wrapper">
                <div id="context-bar" class="hidden"></div>
                
                <div class="input-container">
                    <textarea id="message-input" placeholder="Ask a question or generate code... (Enter to send)"></textarea>
                    
                    <div class="input-actions">
                        <button id="attach-button" class="icon-btn" title="Attach Context">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
                        </button>
                        <button id="send-button" class="icon-btn primary" title="Send">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" x2="11" y1="2" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                        </button>
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