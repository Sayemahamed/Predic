import * as vscode from 'vscode';
import { Model } from '../types';
import { PredicAPIClient } from '../api/client';

export class ModelProvider implements vscode.TreeDataProvider<ModelItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ModelItem | undefined | null | void> = new vscode.EventEmitter<ModelItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<ModelItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private models: Model[] = [];
    private apiClient: PredicAPIClient;
    private pollingIntervals: Map<string, NodeJS.Timeout> = new Map();

    constructor(private context: vscode.ExtensionContext) {
        this.apiClient = new PredicAPIClient();
        this.refresh();
        
        // Refresh every 10 seconds
        setInterval(() => this.refresh(), 10000);
    }

    refresh(): void {
        this.loadModels();
        this._onDidChangeTreeData.fire();
    }

    private async loadModels() {
        try {
            this.models = await this.apiClient.getAvailableModels();
        } catch (error) {
            console.error('Failed to load models:', error);
            this.models = [];
        }
    }

    getTreeItem(element: ModelItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ModelItem): Thenable<ModelItem[]> {
        if (!element) {
            const config = vscode.workspace.getConfiguration('predic');
            const selectedModel = config.get('selectedModel', '');
            
            return Promise.resolve(
                this.models.map(model => new ModelItem(model, selectedModel === model.id))
            );
        }
        return Promise.resolve([]);
    }

    async downloadModel(modelId: string) {
        try {
            await this.apiClient.downloadModel(modelId);
            vscode.window.showInformationMessage(`Started downloading model: ${modelId}`);
            
            // Start polling for status
            this.pollModelStatus(modelId);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to download model: ${error}`);
        }
    }

    private async pollModelStatus(modelId: string) {
        // Clear existing interval if any
        const existingInterval = this.pollingIntervals.get(modelId);
        if (existingInterval) {
            clearInterval(existingInterval);
        }

        const interval = setInterval(async () => {
            try {
                const status = await this.apiClient.getModelStatus(modelId);
                
                // Update model in list
                const modelIndex = this.models.findIndex(m => m.id === modelId);
                if (modelIndex !== -1) {
                    this.models[modelIndex] = { ...this.models[modelIndex], ...status };
                    this.refresh();
                }

                if (status.status === 'ready' || status.status === 'error') {
                    clearInterval(interval);
                    this.pollingIntervals.delete(modelId);
                    
                    if (status.status === 'ready') {
                        vscode.window.showInformationMessage(`Model ${modelId} is ready!`);
                    } else {
                        vscode.window.showErrorMessage(`Failed to download model ${modelId}`);
                    }
                }
            } catch (error) {
                clearInterval(interval);
                this.pollingIntervals.delete(modelId);
                console.error('Failed to poll model status:', error);
            }
        }, 2000);

        this.pollingIntervals.set(modelId, interval);
    }
}

export class ModelItem extends vscode.TreeItem {
    constructor(
        public readonly model: Model,
        private isSelected: boolean
    ) {
        super(model.name, vscode.TreeItemCollapsibleState.None);
        
        this.tooltip = this.getTooltip();
        this.description = this.getDescription();
        this.contextValue = this.model.status;
        this.iconPath = this.getIcon();
        this.command = this.getCommand();
    }

    private getTooltip(): string {
        let tooltip = `${this.model.name}\nSize: ${this.model.size}\nStatus: ${this.model.status}`;
        if (this.isSelected) {
            tooltip += '\n(Currently selected)';
        }
        // Add description if available
        if ((this.model as any).description) {
            tooltip += `\n\n${(this.model as any).description}`;
        }
        return tooltip;
    }

    private getDescription(): string {
        let desc = '';
        const modelWithCategory = this.model as any;
        const sizeCategory = modelWithCategory.size_category || 'medium';
        const categoryLabel = sizeCategory === 'small' ? '🟢' : sizeCategory === 'medium' ? '🟡' : '🔴';
        
        switch (this.model.status) {
            case 'available':
                desc = `${categoryLabel} ${this.model.size} - Click to download`;
                break;
            case 'downloading':
                desc = `${categoryLabel} Downloading... ${Math.round(this.model.progress)}%`;
                break;
            case 'ready':
                desc = this.isSelected ? `${categoryLabel} ${this.model.size} - Selected` : `${categoryLabel} ${this.model.size} - Click to select`;
                break;
            case 'error':
                desc = 'Download failed';
                break;
        }
        return desc;
    }

    private getIcon(): vscode.ThemeIcon {
        if (this.isSelected && this.model.status === 'ready') {
            return new vscode.ThemeIcon('check-all');
        }
        
        switch (this.model.status) {
            case 'available':
                return new vscode.ThemeIcon('cloud-download');
            case 'downloading':
                return new vscode.ThemeIcon('sync~spin');
            case 'ready':
                return new vscode.ThemeIcon('check');
            case 'error':
                return new vscode.ThemeIcon('error');
            default:
                return new vscode.ThemeIcon('question');
        }
    }

    private getCommand(): vscode.Command | undefined {
        if (this.model.status === 'available') {
            return {
                command: 'predic.downloadModel',
                title: 'Download Model',
                arguments: [this]
            };
        } else if (this.model.status === 'ready' && !this.isSelected) {
            return {
                command: 'predic.selectModel',
                title: 'Select Model',
                arguments: [this]
            };
        }
        return undefined;
    }
}