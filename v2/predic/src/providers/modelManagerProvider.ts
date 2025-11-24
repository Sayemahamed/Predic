import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';

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
                case 'downloadEngine':
                    // You can implement the engine downloader here later
                    vscode.window.showInformationMessage("Engine download not yet implemented in this snippet.");
                    break;
            }
        });

        this.panel.onDidDispose(() => { this.panel = undefined; });
        
        // Initial load
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

    private async refreshState() {
        if (!this.panel) return;

        const config = vscode.workspace.getConfiguration('predic');
        const modelDir = config.get<string>('modelDir') || path.join(this.context.extensionUri.fsPath, '..', 'models');
        const activeModel = config.get<string>('modelPath') || '';
        const koboldPath = config.get<string>('koboldCppPath') || '';

        let models: any[] = [];

        if (fs.existsSync(modelDir)) {
            models = fs.readdirSync(modelDir)
                .filter(f => f.endsWith('.gguf'))
                .map(f => {
                    const fullPath = path.join(modelDir, f);
                    return {
                        name: f,
                        path: fullPath,
                        isActive: path.normalize(fullPath) === path.normalize(activeModel)
                    };
                });
        }

        this.panel.webview.postMessage({
            type: 'updateState',
            data: {
                koboldPath,
                modelDir,
                models,
                activeModel
            }
        });
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
            <title>Predic Dashboard</title>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>Predic Dashboard</h1>
                </header>

                <section class="settings-card">
                    <h2>⚙️ Configuration</h2>
                    <div class="setting-item">
                        <label>KoboldCpp Executable</label>
                        <div class="input-group">
                            <input type="text" id="koboldPath" readonly placeholder="Not set (Auto-detecting...)">
                            <button id="btn-kobold-path">Browse...</button>
                        </div>
                    </div>
                    <div class="setting-item">
                        <label>Models Directory</label>
                        <div class="input-group">
                            <input type="text" id="modelDir" readonly placeholder="Not set (Using default)">
                            <button id="btn-model-dir">Browse...</button>
                        </div>
                    </div>
                </section>

                <section class="models-card">
                    <h2>📚 Local Models</h2>
                    <div id="model-list" class="model-list">
                        </div>
                </section>
            </div>
            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}