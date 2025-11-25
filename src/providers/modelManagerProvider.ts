import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';

// --- YOUR MODELS ---
const PREDIC_MODELS = [
    {
        id: 'reacomplete-v1',
        name: 'ReaComplete (Fine-Tuned)',
        description: 'Our official specialized model optimized for React & TypeScript code completion.',
        size: 'Unknown', 
        filename: 'ReaComplete-Q4_K_M.gguf',
        url: 'https://huggingface.co/Sayempro/ReaComplete/resolve/main/ReaComplete-Q4_K_M.gguf' 
    }
];

// --- COMMUNITY MODELS ---
const CURATED_MODELS = [
    {
        id: 'qwen2.5-0.5b',
        name: 'Qwen 2.5 Coder (0.5B)',
        description: 'Ultra-lightweight, fast. Good for older laptops.',
        size: '0.4 GB',
        filename: 'qwen2.5-coder-0.5b-instruct-q4_k_m.gguf',
        url: 'https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf'
    },
    {
        id: 'qwen2.5-1.5b',
        name: 'Qwen 2.5 Coder (1.5B)',
        description: 'Best balance of speed and intelligence for code.',
        size: '1.0 GB',
        filename: 'qwen2.5-coder-1.5b-instruct-q4_k_m.gguf',
        url: 'https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
    },
    {
        id: 'deepseek-1.3b',
        name: 'DeepSeek Coder (1.3B)',
        description: 'Highly capable coding model.',
        size: '0.9 GB',
        filename: 'deepseek-coder-1.3b-instruct.Q4_K_M.gguf',
        url: 'https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.Q4_K_M.gguf'
    },
    {
        id: 'phi-3-mini',
        name: 'Phi-3 Mini (3.8B)',
        description: 'Microsoft\'s powerful small model. Requires 8GB RAM.',
        size: '2.4 GB',
        filename: 'Phi-3-mini-4k-instruct-q4.gguf',
        url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf'
    }
];

export class ModelManagerProvider {
    private panel: vscode.WebviewPanel | undefined;

    constructor(
        private readonly extensionUri: vscode.Uri,
        private context: vscode.ExtensionContext
    ) {}

    public async show() {
        if (this.panel) {
            this.panel.reveal();
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'predicDashboard',
            'Predic Dashboard',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [this.extensionUri]
            }
        );

        this.panel.webview.html = this._getHtmlForWebview(this.panel.webview);
        
        this.panel.webview.onDidReceiveMessage(async message => {
            switch (message.type) {
                case 'refresh':
                    await this.refreshState();
                    break;
                case 'selectPath':
                    await this.handlePathSelection(message.target);
                    break;
                case 'setActiveModel':
                    await this.setActiveModel(message.path);
                    break;
                case 'downloadModel':
                    await this.downloadModel(message.modelId, message.isPredic);
                    break;
            }
        });

        this.panel.onDidDispose(() => { this.panel = undefined; });
        
        await this.refreshState();
    }

    private async handlePathSelection(target: 'koboldCppPath' | 'modelDir') {
        const isFile = target === 'koboldCppPath';
        const uris = await vscode.window.showOpenDialog({
            canSelectFiles: isFile,
            canSelectFolders: !isFile,
            canSelectMany: false,
            title: isFile ? "Select KoboldCpp Executable" : "Select Models Directory"
        });

        if (uris && uris[0]) {
            await vscode.workspace.getConfiguration('predic').update(target, uris[0].fsPath, vscode.ConfigurationTarget.Global);
            await this.refreshState();
        }
    }

    private async setActiveModel(modelPath: string) {
        await vscode.workspace.getConfiguration('predic').update('modelPath', modelPath, vscode.ConfigurationTarget.Global);
        vscode.commands.executeCommand('predic.restartServer');
        vscode.window.showInformationMessage(`Switched model. Restarting server...`);
        await this.refreshState();
    }

    private async downloadModel(modelId: string, isPredic: boolean) {
        const list = isPredic ? PREDIC_MODELS : CURATED_MODELS;
        const model = list.find(m => m.id === modelId);
        
        if (!model) return;

        const config = vscode.workspace.getConfiguration('predic');
        let modelDir = config.get<string>('modelDir');
        if (!modelDir) {
             modelDir = path.join(this.context.extensionUri.fsPath, '..', 'models');
        }

        if (!fs.existsSync(modelDir)) {
            fs.mkdirSync(modelDir, { recursive: true });
        }

        const destPath = path.join(modelDir, model.filename);

        if (fs.existsSync(destPath)) {
            vscode.window.showInformationMessage(`Model ${model.filename} already exists!`);
            return;
        }

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Downloading ${model.name}...`,
            cancellable: true
        }, (progress, token) => {
            return new Promise<void>((resolve, reject) => {
                const file = fs.createWriteStream(destPath);
                const request = https.get(model.url, (response) => {
                    if (response.statusCode === 301 || response.statusCode === 302) {
                        // Handle redirects if necessary
                    }

                    const totalLen = parseInt(response.headers['content-length'] || '0', 10);
                    let curLen = 0;

                    response.pipe(file);

                    response.on('data', (chunk) => {
                        curLen += chunk.length;
                        if (totalLen > 0) {
                            const percentage = (curLen / totalLen) * 100;
                            progress.report({ message: `${percentage.toFixed(1)}%` });
                        } else {
                            progress.report({ message: `${(curLen / 1024 / 1024).toFixed(1)} MB` });
                        }
                        if (token.isCancellationRequested) {
                            response.destroy();
                            file.destroy();
                            fs.unlink(destPath, () => {});
                            reject(new Error("Cancelled"));
                        }
                    });

                    file.on('finish', () => {
                        file.close();
                        resolve();
                    });
                });

                request.on('error', (err) => {
                    fs.unlink(destPath, () => {});
                    reject(err);
                });
            });
        }).then(() => {
            vscode.window.showInformationMessage(`Downloaded ${model.name} successfully!`);
            this.refreshState();
        }, (err) => {
            if (err.message !== "Cancelled") {
                vscode.window.showErrorMessage(`Download failed: ${err.message}`);
            }
        });
    }

    private async refreshState() {
        if (!this.panel) return;

        const config = vscode.workspace.getConfiguration('predic');
        const modelDir = config.get<string>('modelDir') || path.join(this.context.extensionUri.fsPath, '..', 'models');
        const activeModel = config.get<string>('modelPath') || '';
        const koboldPath = config.get<string>('koboldCppPath') || '';

        let localFiles: string[] = [];

        if (fs.existsSync(modelDir)) {
            localFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.gguf'));
        }

        const localModels = localFiles.map(f => {
            const fullPath = path.join(modelDir, f);
            return {
                name: f,
                path: fullPath,
                isActive: path.normalize(fullPath) === path.normalize(activeModel)
            };
        });

        const mapModel = (m: any) => {
            const isDownloaded = localFiles.includes(m.filename);
            const fullPath = path.join(modelDir, m.filename);
            return {
                ...m,
                isDownloaded: isDownloaded,
                isActive: isDownloaded && (path.normalize(fullPath) === path.normalize(activeModel))
            };
        };

        const curatedModels = CURATED_MODELS.map(mapModel);
        const predicModels = PREDIC_MODELS.map(mapModel);

        this.panel.webview.postMessage({
            type: 'updateState',
            data: {
                koboldPath,
                modelDir,
                localModels,
                curatedModels,
                predicModels,
                activeModel
            }
        });
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'styles', 'modelManager.css'));
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'scripts', 'modelManager.js'));
        const logoUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'assets', 'background.png'));
        
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} https: data:; style-src ${webview.cspSource} 'unsafe-inline'; script-src ${webview.cspSource};">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>Predic Dashboard</title>
        </head>
        <body>
            <div class="container">
                <header class="dashboard-header">
                    <div class="logo-container">
                        <img src="${logoUri}" alt="Predic Logo" class="dashboard-logo">
                        <div>
                            <h1>Predic Dashboard</h1>
                            <p class="subtitle">Manage your local AI engine and models</p>
                        </div>
                    </div>
                </header>

                <section class="models-card special-card">
                    <h2>🏆 Official Predic Models</h2>
                    <p class="section-desc">Try our own fine-tuned models optimized specifically for this extension.</p>
                    <div id="predic-list" class="model-grid"></div>
                </section>

                <section class="models-card">
                    <h2>⭐ Community Recommended</h2>
                    <div id="curated-list" class="model-grid"></div>
                </section>

                <section class="models-card">
                    <h2>📂 Your Local Models</h2>
                    <div id="local-list" class="model-list"></div>
                </section>

                <section class="settings-card">
                    <h2>⚙️ Configuration</h2>
                    <div class="setting-item">
                        <label>KoboldCpp Executable</label>
                        <div class="input-group">
                            <input type="text" id="koboldPath" readonly placeholder="Auto-detecting...">
                            <button id="btn-kobold-path" class="btn-secondary">Browse...</button>
                        </div>
                    </div>
                    <div class="setting-item">
                        <label>Models Directory</label>
                        <div class="input-group">
                            <input type="text" id="modelDir" readonly placeholder="Using default...">
                            <button id="btn-model-dir" class="btn-secondary">Browse...</button>
                        </div>
                    </div>
                </section>
            </div>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}