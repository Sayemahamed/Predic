import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import { Model } from '../types';

export class ModelManagerProvider {
    private panel: vscode.WebviewPanel | undefined;
    
    // Define curated models with their URLs
    private availableModels: Model[] = [
        {
            id: 'qwen2.5-coder-0.5b',
            name: 'Qwen 2.5 Coder (0.5B)',
            size: '0.5 GB',
            status: 'missing',
            url: 'https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf'
        },
        {
            id: 'deepseek-coder-1.3b',
            name: 'DeepSeek Coder (1.3B)',
            size: '1.3 GB',
            status: 'missing',
            url: 'https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.Q4_K_M.gguf'
        }
    ];

    constructor(
        private readonly extensionUri: vscode.Uri,
        private context: vscode.ExtensionContext
    ) {}

    public async show() {
        const column = vscode.window.activeTextEditor ? vscode.window.activeTextEditor.viewColumn : undefined;
        if (this.panel) {
            this.panel.reveal(column);
            return;
        }

        this.panel = vscode.window.createWebviewPanel('predicModelManager', 'Predic Models', column || vscode.ViewColumn.One, {
            enableScripts: true,
            localResourceRoots: [this.extensionUri]
        });

        this.panel.webview.html = this._getHtmlForWebview(this.panel.webview);
        
        this.panel.webview.onDidReceiveMessage(async message => {
            switch (message.type) {
                case 'webviewReady':
                case 'refreshModels':
                    await this.refreshModels();
                    break;
                case 'downloadModel':
                    await this.downloadModel(message.modelId);
                    break;
                case 'selectModel':
                    await this.selectModel(message.modelId);
                    break;
            }
        });

        this.panel.onDidDispose(() => { this.panel = undefined; });
    }

    private getModelsDirectory(): string {
        // Assuming v2 structure: workspace/models
        // Adjust this path logic based on where you want to store models
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
        return workspaceRoot ? path.join(workspaceRoot, '..', 'models') : '';
    }

    private async refreshModels() {
        const modelsDir = this.getModelsDirectory();
        if (!modelsDir || !fs.existsSync(modelsDir)) {
            // Send curated list with 'missing' status
            this.panel?.webview.postMessage({ type: 'updateModels', models: this.availableModels });
            return;
        }

        const files = fs.readdirSync(modelsDir);
        
        // Update status based on files on disk
        const updatedModels = this.availableModels.map(model => {
            const fileName = path.basename(model.url || '');
            const exists = files.includes(fileName);
            return { ...model, status: exists ? 'ready' : 'missing' };
        });

        this.panel?.webview.postMessage({ type: 'updateModels', models: updatedModels });
    }

    private async downloadModel(modelId: string) {
        const model = this.availableModels.find(m => m.id === modelId);
        if (!model || !model.url) return;

        const modelsDir = this.getModelsDirectory();
        if (!fs.existsSync(modelsDir)) {
            fs.mkdirSync(modelsDir, { recursive: true });
        }

        const fileName = path.basename(model.url);
        const destPath = path.join(modelsDir, fileName);

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Downloading ${model.name}...`,
            cancellable: false
        }, async (progress) => {
            return new Promise<void>((resolve, reject) => {
                const file = fs.createWriteStream(destPath);
                https.get(model.url!, (response) => {
                    const totalLen = parseInt(response.headers['content-length'] || '0');
                    let curLen = 0;

                    response.pipe(file);
                    
                    response.on('data', (chunk) => {
                        curLen += chunk.length;
                        const percentage = Math.round((curLen / totalLen) * 100);
                        progress.report({ message: `${percentage}%` });
                    });

                    file.on('finish', () => {
                        file.close();
                        vscode.window.showInformationMessage(`${model.name} downloaded successfully!`);
                        this.refreshModels();
                        resolve();
                    });
                }).on('error', (err) => {
                    fs.unlink(destPath, () => {}); // Delete partial file
                    vscode.window.showErrorMessage(`Download failed: ${err.message}`);
                    reject(err);
                });
            });
        });
    }

    private async selectModel(modelId: string) {
        const model = this.availableModels.find(m => m.id === modelId);
        if(!model || !model.url) return;
        
        const modelsDir = this.getModelsDirectory();
        const fileName = path.basename(model.url);
        const fullPath = path.join(modelsDir, fileName);

        // 1. Update VS Code settings
        await vscode.workspace.getConfiguration('predic').update('modelPath', fullPath, vscode.ConfigurationTarget.Global);
        
        // 2. Trigger Programmatic Restart
        vscode.commands.executeCommand('predic.restartServer');
        
        vscode.window.showInformationMessage(`Switched to ${model.name}`);
        
        // 3. Refresh list to show new active status
        this.refreshModels(); 
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'modelManager.css'));
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'modelManager.js'));
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>Models</title>
        </head>
        <body>
            <div class="container">
                <h1>Predic Models</h1>
                <div id="modelList"></div>
            </div>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}