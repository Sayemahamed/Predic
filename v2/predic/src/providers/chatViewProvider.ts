import * as vscode from 'vscode';
import { PredicAPIClient } from '../api/client';

export class ChatViewProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private apiClient: PredicAPIClient;

    constructor(private readonly _extensionUri: vscode.Uri) {
        this.apiClient = new PredicAPIClient();
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        // 1. Allow Scripts
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        // 2. Set HTML
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // 3. Handle Messages from Frontend
        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'sendMessage':
                    await this.handleUserMessage(data.value);
                    break;
            }
        });
    }

    private async handleUserMessage(content: string) {
        if (!this._view) return;

        // 1. Echo User Message
        this._view.webview.postMessage({ type: 'addMessage', role: 'user', content: content });

        try {
            // 2. Call KoboldCpp
            // Note: If you see "Thinking..." indefinitely, the model might be slow or config wrong.
            const messages = [{ role: "user", content: content }];
            const response = await this.apiClient.chat(messages);
            
            // 3. Send AI Response
            const aiText = response.data.choices[0].message.content;
            this._view.webview.postMessage({ type: 'addMessage', role: 'assistant', content: aiText });

        } catch (error: any) {
            this._view.webview.postMessage({ 
                type: 'addMessage', 
                role: 'system', 
                content: `Error: ${error.message || "Failed to connect to KoboldCpp"}` 
            });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.css'));
        const nonce = getNonce();

        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>Predic Chat</title>
        </head>
        <body>
            <div id="chat-container"></div>
            <div class="input-area">
                <textarea id="message-input" placeholder="Ask about your code..."></textarea>
                <button id="send-button">Send</button>
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