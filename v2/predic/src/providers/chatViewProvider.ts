import * as vscode from 'vscode';
import { PredicAPIClient } from '../api/client';
import { ChatMessage } from '../types';

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
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'sendMessage':
                    await this.handleMessage(data.message);
                    break;
                case 'webviewReady':
                    // No need to fetch models list for chat anymore
                    break;
            }
        });
    }

    private async handleMessage(content: string) {
        if (!this._view) return;

        // 1. Send user message back to UI to display immediately
        this._view.webview.postMessage({
            type: 'addMessage',
            message: { role: 'user', content: content } as ChatMessage
        });

        try {
            // 2. Call API
            const messages = [{ role: "user", content: content }];
            const response = await this.apiClient.chat(messages);
            
            // 3. Extract actual text from OpenAI response format
            const aiText = response.data.choices[0].message.content;

            // 4. Send AI response to UI
            this._view.webview.postMessage({
                type: 'addMessage',
                message: { role: 'assistant', content: aiText } as ChatMessage
            });

        } catch (error) {
            this._view.webview.postMessage({
                type: 'addMessage',
                message: { role: 'system', content: `Error: ${error}` } as ChatMessage
            });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        // Keep your existing HTML generation logic here (copy from your file)
        // Just ensure it points to the correct script/css paths
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.css'));
        
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>Predic Chat</title>
        </head>
        <body>
            <div id="chat-container"></div>
            <div id="input-container">
                <textarea id="message-input" placeholder="Ask Predic..."></textarea>
                <button id="send-button">Send</button>
            </div>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}