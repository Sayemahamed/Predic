import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export class ModelProvider implements vscode.TreeDataProvider<ModelItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ModelItem | undefined | null | void> = new vscode.EventEmitter<ModelItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<ModelItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor() {
        // Refresh tree when configuration changes (e.g. user selects a new model)
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('predic.modelPath')) {
                this.refresh();
            }
        });
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: ModelItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ModelItem): Thenable<ModelItem[]> {
        if (element) {
            return Promise.resolve([]);
        }

        const workspaceRoot = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
        const modelsDir = workspaceRoot ? path.join(workspaceRoot, '..', 'models') : '';
        
        // Get currently active model path from config
        const config = vscode.workspace.getConfiguration('predic');
        const currentModelPath = config.get<string>('modelPath') || '';

        if (!modelsDir || !fs.existsSync(modelsDir)) {
            return Promise.resolve([new ModelItem('No models folder found', vscode.TreeItemCollapsibleState.None, false)]);
        }

        const files = fs.readdirSync(modelsDir).filter(f => f.endsWith('.gguf'));
        
        if (files.length === 0) {
            return Promise.resolve([new ModelItem('No models downloaded', vscode.TreeItemCollapsibleState.None, false)]);
        }

        return Promise.resolve(files.map(file => {
            const fullPath = path.join(modelsDir, file);
            // Check if this file is the active one
            // We normalize paths to ensure cross-platform compatibility (Windows vs Unix slashes)
            const isActive = path.normalize(fullPath) === path.normalize(currentModelPath);
            return new ModelItem(file, vscode.TreeItemCollapsibleState.None, isActive);
        }));
    }
}

class ModelItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly isActive: boolean
    ) {
        super(label, collapsibleState);
        this.tooltip = this.label;
        
        // Visual customization for active/inactive states
        if (this.isActive) {
            this.description = "(Active)";
            this.iconPath = new vscode.ThemeIcon('check', new vscode.ThemeColor('charts.green'));
            this.contextValue = 'activeModel';
        } else {
            this.description = "GGUF";
            this.iconPath = new vscode.ThemeIcon('database');
            this.contextValue = 'model';
        }
    }
}