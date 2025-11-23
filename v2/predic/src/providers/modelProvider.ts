import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export class ModelProvider implements vscode.TreeDataProvider<ModelItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ModelItem | undefined | null | void> = new vscode.EventEmitter<ModelItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<ModelItem | undefined | null | void> = this._onDidChangeTreeData.event;

    constructor() {}

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

        // Get models directory (Same logic as Manager)
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0].uri.fsPath;
        const modelsDir = workspaceRoot ? path.join(workspaceRoot, '..', 'models') : '';

        if (!modelsDir || !fs.existsSync(modelsDir)) {
            return Promise.resolve([new ModelItem('No models folder found', vscode.TreeItemCollapsibleState.None)]);
        }

        const files = fs.readdirSync(modelsDir).filter(f => f.endsWith('.gguf'));
        
        if (files.length === 0) {
            return Promise.resolve([new ModelItem('No models downloaded', vscode.TreeItemCollapsibleState.None)]);
        }

        return Promise.resolve(files.map(file => new ModelItem(file, vscode.TreeItemCollapsibleState.None)));
    }
}

class ModelItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(label, collapsibleState);
        this.tooltip = this.label;
        this.description = "Local GGUF";
        this.iconPath = new vscode.ThemeIcon('database');
    }
}