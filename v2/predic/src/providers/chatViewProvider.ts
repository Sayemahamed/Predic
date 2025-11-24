import * as vscode from 'vscode';
import * as fs from 'fs';
import { PredicAPIClient } from '../api/client';

export class ChatViewProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private apiClient: PredicAPIClient;
    // We keep track of context to avoid sending it repeatedly if not needed
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

    // Helper for the Right-Click commands
    public async triggerAnalysis(prompt: string, codeContext: string) {
        if (this._view) {
            this._view.show?.(true);
            this.activeContext = codeContext;
            this.activeFileName = "Selected Code";
            
            // Show the user what we are doing
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
            filters: { 'Code': ['ts', 'js', 'py', 'java', 'c', 'cpp', 'json', 'md', 'txt'] }
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
            vscode.window.showInformationMessage('Code accepted!');
        } else {
            vscode.window.showErrorMessage('Open a file to accept this code.');
        }
    }

    private async parseAndLoadFileRefs(content: string): Promise<string> {
        // Regex to find @filename
        const fileRegex = /@([a-zA-Z0-9_\-\.]+)/g;
        const matches = [...content.matchAll(fileRegex)];
        let loadedContext = "";

        for (const match of matches) {
            const fileName = match[1];
            // Search for file in workspace
            const files = await vscode.workspace.findFiles(`**/${fileName}`, '**/node_modules/**', 1);
            if (files.length > 0) {
                try {
                    const fileContent = fs.readFileSync(files[0].fsPath, 'utf-8');
                    loadedContext += `\nContent of ${fileName}:\n\`\`\`\n${fileContent}\n\`\`\`\n`;
                    // Visual feedback
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

        // 1. Show User Message
        this._view.webview.postMessage({ type: 'addMessage', role: 'user', content: content });

        try {
            // 2. Handle @file_name parsing
            const dynamicContext = await this.parseAndLoadFileRefs(content);
            
            // 3. Construct Prompt
            let fullPrompt = "";
            
            // Add manually attached context
            if (this.activeContext) {
                fullPrompt += `Context (${this.activeFileName}):\n\`\`\`\n${this.activeContext}\n\`\`\`\n`;
            }
            
            // Add @file context
            if (dynamicContext) {
                fullPrompt += `\nReferenced Files:\n${dynamicContext}\n`;
            }

            fullPrompt += `\nUser Query: ${content}`;

            // 4. Send to AI
            const messages = [
                { 
                    role: "system", 
                    content: "You are Predic. Provide ONLY code or direct answers. Do NOT explain code unless explicitly asked. Do NOT add conversational filler." 
                },
                { role: "user", content: fullPrompt }
            ];
            
            const response = await this.apiClient.chat(messages);
            const aiText = response.data.choices[0].message.content;
            
            this._view.webview.postMessage({ type: 'addMessage', role: 'assistant', content: aiText });
            
            // Clear manual context after use so it doesn't pollute future chats
            if(this.activeContext) {
                this.activeContext = "";
                this._view.webview.postMessage({ type: 'clearContext' });
            }

        } catch (error: any) {
            this._view.webview.postMessage({ 
                type: 'addMessage', 
                role: 'system', 
                content: `Error: ${error.message || "Connection failed."}` 
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
            <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>Predic Chat</title>
        </head>
        <body>
            <div id="chat-container"></div>
            
            <div id="context-indicator" class="hidden">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M13.5 3.5L12.5 2.5L8 7L3.5 2.5L2.5 3.5L7 8L2.5 12.5L3.5 13.5L8 9L12.5 13.5L13.5 12.5L9 8L13.5 3.5Z"/></svg>
                <span id="context-filename"></span>
            </div>

            <div class="input-area">
                <button id="attach-button" class="icon-btn" title="Attach File">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                        <path d="M10.57 2.57L11.28 3.28C12.64 4.65 12.64 6.86 11.28 8.22L6.83 12.67C5.79 13.71 4.1 13.71 3.06 12.67L2.57 12.18C1.53 11.14 1.53 9.45 2.57 8.41L7.02 3.96C7.29 3.69 7.72 3.69 7.99 3.96L8.47 4.44C8.74 4.71 8.74 5.14 8.47 5.41L4.02 9.86C3.78 10.1 3.78 10.49 4.02 10.73L4.51 11.22C4.75 11.46 5.14 11.46 5.38 11.22L9.83 6.77C10.39 6.21 10.39 5.31 9.83 4.75L9.12 4.04C8.56 3.48 7.66 3.48 7.1 4.04L2.65 8.49C2.41 8.73 2.41 9.12 2.65 9.36C2.89 9.6 3.28 9.6 3.52 9.36L7.97 4.91L8.68 5.62L4.23 10.07C4.23 10.07 4.23 10.07 4.23 10.07C3.59 10.71 3.59 11.74 4.23 12.38L4.72 12.87C5.36 13.51 6.39 13.51 7.03 12.87L11.48 8.42C12.12 7.78 12.12 6.75 11.48 6.11L10.77 5.4C10.13 4.76 9.1 4.76 8.46 5.4L7.51 6.35L6.8 5.64L7.75 4.69C8.8 3.64 10.49 3.64 11.53 4.69C11.53 4.69 11.53 4.69 11.53 4.69L12.24 5.4C13.28 6.45 13.28 8.14 12.24 9.19L7.79 13.64C5.69 15.74 2.29 15.74 0.19 13.64L0.19 13.64C-1.91 11.54 -1.91 8.14 0.19 6.04L4.64 1.59L5.35 2.3L0.9 6.75C-0.81 8.46 -0.81 11.23 0.9 12.94L1.39 13.43C3.1 15.14 5.87 15.14 7.58 13.43L12.03 8.98C12.67 8.34 12.67 7.31 12.03 6.67L11.32 5.96C10.68 5.32 9.65 5.32 9.01 5.96L8.06 6.91L7.35 6.2L8.3 5.25C9.34 4.2 11.03 4.2 12.07 5.25L12.78 5.96C13.82 7.01 13.82 8.7 12.78 9.75L8.33 14.2C6.23 16.3 2.83 16.3 0.73 14.2L0.24 13.71C-1.86 11.61 -1.86 8.21 0.24 6.11L4.69 1.66C5.33 1.02 6.36 1.02 7 1.66C7.64 2.3 7.64 3.33 7 3.97L6.51 4.46L5.8 3.75L6.29 3.26C6.53 3.02 6.53 2.63 6.29 2.39C6.05 2.15 5.66 2.15 5.42 2.39L0.97 6.84C-0.74 8.55 -0.74 11.32 0.97 13.03L1.46 13.52C3.17 15.23 5.94 15.23 7.65 13.52L12.1 9.07C12.74 8.43 12.74 7.4 12.1 6.76L11.39 6.05C10.75 5.41 9.72 5.41 9.08 6.05L8.13 7L7.42 6.29L8.37 5.34C9.41 4.29 11.1 4.29 12.14 5.34L12.85 6.05C13.89 7.1 13.89 8.79 12.85 9.84L8.4 14.29C6.3 16.39 2.9 16.39 0.8 14.29L0.31 13.8C-1.79 11.7 -1.79 8.3 0.31 6.2L4.76 1.75L5.47 2.46L1.02 6.91C-0.69 8.62 -0.69 11.39 1.02 13.1L1.51 13.59C3.22 15.3 5.99 15.3 7.7 13.59L12.15 9.14C12.79 8.5 12.79 7.47 12.15 6.83L11.44 6.12C10.8 5.48 9.77 5.48 9.13 6.12L8.18 7.07L7.47 6.36L8.42 5.41C9.46 4.36 11.15 4.36 12.19 5.41L12.9 6.12C13.94 7.17 13.94 8.86 12.9 9.91L8.45 14.36C6.35 16.46 2.95 16.46 0.85 14.36L0.36 13.87C-1.74 11.77 -1.74 8.37 0.36 6.27L4.81 1.82L5.52 2.53L1.07 6.98C-0.64 8.69 -0.64 11.46 1.07 13.17L1.56 13.66C3.27 15.37 6.04 15.37 7.75 13.66L12.2 9.21C12.84 8.57 12.84 7.54 12.2 6.9L11.49 6.19C10.85 5.55 9.82 5.55 9.18 6.19L8.23 7.14L7.52 6.43L8.47 5.48C9.51 4.43 11.2 4.43 12.24 5.48L12.95 6.19C13.99 7.24 13.99 8.93 12.95 9.98L8.5 14.43C6.4 16.53 3 16.53 0.9 14.43L0.41 13.94C-1.69 11.84 -1.69 8.44 0.41 6.34L4.86 1.89L5.57 2.6L1.12 7.05C-0.59 8.76 -0.59 11.53 1.12 13.24L1.61 13.73C3.32 15.44 6.09 15.44 7.8 13.73L12.25 9.28C12.89 8.64 12.89 7.61 12.25 6.97L11.54 6.26C10.9 5.62 9.87 5.62 9.23 6.26L8.28 7.21L7.57 6.5L8.52 5.55C9.56 4.5 11.25 4.5 12.29 5.55L13 6.26C14.04 7.31 14.04 9 13 10.05L8.55 14.5C6.45 16.6 3.05 16.6 0.95 14.5L0.46 14.01C-1.64 11.91 -1.64 8.51 0.46 6.41L4.91 1.96L5.62 2.67L1.17 7.12C-0.54 8.83 -0.54 11.6 1.17 13.31L1.66 13.8C3.37 15.51 6.14 15.51 7.85 13.8L12.3 9.35C12.94 8.71 12.94 7.68 12.3 7.04L11.59 6.33C10.95 5.69 9.92 5.69 9.28 6.33L8.33 7.28L7.62 6.57L8.57 5.62C9.61 4.57 11.3 4.57 12.34 5.62L13.05 6.33C14.09 7.38 14.09 9.07 13.05 10.12L8.6 14.57C6.5 16.67 3.1 16.67 1 14.57L0.51 14.08C-1.59 11.98 -1.59 8.58 0.51 6.48L4.96 2.03L5.67 2.74L1.22 7.19C-0.49 8.9 -0.49 11.67 1.22 13.38L1.71 13.87C3.42 15.58 6.19 15.58 7.9 13.87L12.35 9.42C12.99 8.78 12.99 7.75 12.35 7.11L11.64 6.4C11 5.76 9.97 5.76 9.33 6.4L8.38 7.35L7.67 6.64L8.62 5.69C9.66 4.64 11.35 4.64 12.39 5.69L13.1 6.4C14.14 7.45 14.14 9.14 13.1 10.19L8.65 14.64C6.55 16.74 3.15 16.74 1.05 14.64L0.56 14.15C-1.54 12.05 -1.54 8.65 0.56 6.55L5.01 2.1L5.72 2.81L1.27 7.26C-0.44 8.97 -0.44 11.74 1.27 13.45L1.76 13.94C3.47 15.65 6.24 15.65 7.95 13.94L12.4 9.49C13.04 8.85 13.04 7.82 12.4 7.18L11.69 6.47C11.05 5.83 10.02 5.83 9.38 6.47L8.43 7.42L7.72 6.71L8.67 5.76C9.71 4.71 11.4 4.71 12.44 5.76L13.15 6.47C14.19 7.52 14.19 9.21 13.15 10.26L8.7 14.71C6.6 16.81 3.2 16.81 1.1 14.71L0.61 14.22C-1.49 12.12 -1.49 8.72 0.61 6.62L5.06 2.17L5.77 2.88L1.32 7.33C-0.39 9.04 -0.39 11.81 1.32 13.52L1.81 14.01C3.52 15.72 6.29 15.72 8 14.01L12.45 9.56C13.09 8.92 13.09 7.89 12.45 7.25L11.74 6.54C11.1 5.9 10.07 5.9 9.43 6.54L8.48 7.49L7.77 6.78L8.72 5.83C9.76 4.78 11.45 4.78 12.49 5.83L13.2 6.54C14.24 7.59 14.24 9.28 13.2 10.33L8.75 14.78C6.65 16.88 3.25 16.88 1.15 14.78L0.66 14.29C-1.44 12.19 -1.44 8.79 0.66 6.69L5.11 2.24L5.82 2.95L1.37 7.4C-0.34 9.11 -0.34 11.88 1.37 13.59L1.86 14.08C3.57 15.79 6.34 15.79 8.05 14.08L12.5 9.63C13.14 8.99 13.14 7.96 12.5 7.32L11.79 6.61C11.15 5.97 10.12 5.97 9.48 6.61L8.53 7.56L7.82 6.85L8.77 5.9C9.81 4.85 11.5 4.85 12.54 5.9L13.25 6.61C14.29 7.66 14.29 9.35 13.25 10.4L8.8 14.85C6.7 16.95 3.3 16.95 1.2 14.85L0.71 14.36C-1.39 12.26 -1.39 8.86 0.71 6.76L5.16 2.31L5.87 3.02L1.42 7.47C-0.29 9.18 -0.29 11.95 1.42 13.66L1.91 14.15C3.62 15.86 6.39 15.86 8.1 14.15L12.55 9.7C13.19 9.06 13.19 8.03 12.55 7.39L11.84 6.68C11.2 6.04 10.17 6.04 9.53 6.68L8.58 7.63L7.87 6.92L8.82 5.97C9.86 4.92 11.55 4.92 12.59 5.97L13.3 6.68C14.34 7.73 14.34 9.42 13.3 10.47L8.85 14.92C6.75 17.02 3.35 17.02 1.25 14.92L0.76 14.43C-1.34 12.33 -1.34 8.93 0.76 6.83L5.21 2.38L5.92 3.09L1.47 7.54C-0.24 9.25 -0.24 12.02 1.47 13.73L1.96 14.22C3.67 15.93 6.44 15.93 8.15 14.22L12.6 9.77C13.24 9.13 13.24 8.1 12.6 7.46L11.89 6.75C11.25 6.11 10.22 6.11 9.58 6.75L8.63 7.7L7.92 6.99L8.87 6.04C9.91 4.99 11.6 4.99 12.64 6.04L13.35 6.75C14.39 7.8 14.39 9.49 13.35 10.54L8.9 14.99C6.8 17.09 3.4 17.09 1.3 14.99L0.81 14.5C-1.29 12.4 -1.29 9 0.81 6.9L5.26 2.45L5.97 3.16ZM1.5 3.5L2.5 2.5L7 7L2.5 11.5L1.5 10.5L6 6L1.5 1.5L2.5 0.5L7 5L11.5 0.5L12.5 1.5L8 6L12.5 10.5L11.5 11.5L7 7L2.5 2.5L1.5 3.5Z" transform="scale(0.5) translate(10,10)"/></svg>
                </button>
                <textarea id="message-input" placeholder="Ask Predic (use @filename for context)..."></textarea>
                <button id="send-button" class="icon-btn primary" title="Send">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M1.1 8L15 1L8 15L6.5 9.5L1.1 8Z"/></svg>
                </button>
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