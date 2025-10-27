import * as vscode from 'vscode';
import { ModelProvider } from './providers/modelProvider';
import { ChatViewProvider } from './providers/chatViewProvider';
import { ModelManagerProvider } from './providers/modelManagerProvider';
import { ServerManager } from './serverManager';
import { PredicAPIClient } from './api/client';

let serverManager: ServerManager;
let modelManagerProvider: ModelManagerProvider;

export async function activate(context: vscode.ExtensionContext) {
    console.log('Predic extension is now active!');

    // Initialize server manager
    serverManager = new ServerManager(context);
    
    // Initialize API client
    const apiClient = new PredicAPIClient();

    // Initialize providers
    const modelProvider = new ModelProvider(context);
    modelManagerProvider = new ModelManagerProvider(context.extensionUri, context);
    const chatViewProvider = new ChatViewProvider(context.extensionUri, context);

    // Register tree data provider
    vscode.window.registerTreeDataProvider('predic.models', modelProvider);

    // Register webview view provider for chat in panel
    vscode.window.registerWebviewViewProvider(
        ChatViewProvider.viewType,
        chatViewProvider,
        {
            webviewOptions: {
                retainContextWhenHidden: true
            }
        }
    );

    // Register commands
    const openModelManagerCommand = vscode.commands.registerCommand(
        'predic.openModelManager',
        () => {
            modelManagerProvider.show();
        }
    );

    const openChatCommand = vscode.commands.registerCommand(
        'predic.openChat',
        () => {
            // Focus on the chat panel
            vscode.commands.executeCommand('workbench.view.extension.predic-chat');
        }
    );

    const startServerCommand = vscode.commands.registerCommand(
        'predic.startServer',
        async () => {
            await serverManager.start();
        }
    );

    const stopServerCommand = vscode.commands.registerCommand(
        'predic.stopServer',
        async () => {
            await serverManager.stop();
        }
    );

    const refreshModelsCommand = vscode.commands.registerCommand(
        'predic.refreshModels',
        () => {
            modelProvider.refresh();
        }
    );

    // Register model tree item click handlers
    const downloadModelCommand = vscode.commands.registerCommand(
        'predic.downloadModel',
        async (item: any) => {
            if (item && item.model) {
                await modelProvider.downloadModel(item.model.id);
            }
        }
    );

    const selectModelCommand = vscode.commands.registerCommand(
        'predic.selectModel',
        async (item: any) => {
            if (item && item.model) {
                const config = vscode.workspace.getConfiguration('predic');
                await config.update('selectedModel', item.model.id, vscode.ConfigurationTarget.Global);
                vscode.window.showInformationMessage(`Selected model: ${item.model.name}`);
                modelProvider.refresh();
            }
        }
    );

    // Test connection command
    const testConnectionCommand = vscode.commands.registerCommand(
        'predic.testConnection',
        async () => {
            const apiClient = new PredicAPIClient();
            try {
                const healthy = await apiClient.checkHealth();
                if (healthy) {
                    const models = await apiClient.getAvailableModels();
                    vscode.window.showInformationMessage(
                        `Server is healthy! Found ${models.length} available models.`
                    );
                } else {
                    vscode.window.showErrorMessage('Server is not responding');
                }
            } catch (error) {
                vscode.window.showErrorMessage(`Connection failed: ${error}`);
            }
        }
    );

    // Register server status provider
    const serverStatusProvider = new ServerStatusProvider(serverManager);
    vscode.window.registerTreeDataProvider('predic.server', serverStatusProvider);

    context.subscriptions.push(
        openModelManagerCommand,
        openChatCommand,
        startServerCommand,
        stopServerCommand,
        refreshModelsCommand,
        downloadModelCommand,
        selectModelCommand,
        testConnectionCommand
    );

    // Modified auto-start server logic
    const config = vscode.workspace.getConfiguration('predic');
    if (config.get('autoStartServer', true)) {
        // Don't auto-start, just check if it's running
        const isHealthy = await serverManager.isServerHealthy();
        if (isHealthy) {
            vscode.window.showInformationMessage('Predic server is already running');
        } else {
            const choice = await vscode.window.showInformationMessage(
                'Predic server is not running. Would you like to start it?',
                'Start Server',
                'Use External Server'
            );
            
            if (choice === 'Start Server') {
                await serverManager.start();
            }
        }
    }

    // Show welcome message
    const showWelcome = context.globalState.get('predic.showWelcome', true);
    if (showWelcome) {
        const selection = await vscode.window.showInformationMessage(
            'Welcome to Predic! Your offline AI coding assistant is ready.',
            'Open Chat',
            'Manage Models',
            'Test Connection',
            'Don\'t show again'
        );
        
        if (selection === 'Open Chat') {
            vscode.commands.executeCommand('predic.openChat');
        } else if (selection === 'Manage Models') {
            vscode.commands.executeCommand('predic.openModelManager');
        } else if (selection === 'Test Connection') {
            vscode.commands.executeCommand('predic.testConnection');
        } else if (selection === 'Don\'t show again') {
            context.globalState.update('predic.showWelcome', false);
        }
    }
}

export function deactivate() {
    if (serverManager) {
        serverManager.stop();
    }
}

// Server Status Provider for the sidebar
class ServerStatusProvider implements vscode.TreeDataProvider<ServerStatusItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ServerStatusItem | undefined | null | void> = new vscode.EventEmitter<ServerStatusItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<ServerStatusItem | undefined | null | void> = this._onDidChangeTreeData.event;
    
    private isHealthy: boolean = false;
    private isRunning: boolean = false;

    constructor(private serverManager: ServerManager) {
        // Listen to server status changes
        serverManager.onStatusChanged((healthy) => {
            this.isHealthy = healthy;
            this.isRunning = this.serverManager.isRunning();
            this.refresh();
        });
        
        // Initial check
        this.checkStatus();
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    private async checkStatus() {
        this.isHealthy = await this.serverManager.isServerHealthy();
        this.isRunning = this.serverManager.isRunning();
        this.refresh();
    }

    getTreeItem(element: ServerStatusItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: ServerStatusItem): Promise<ServerStatusItem[]> {
        if (!element) {
            // Re-check status when tree is refreshed
            await this.checkStatus();

            return [
                new ServerStatusItem(
                    'Status',
                    this.isRunning ? 'Running' : 'Stopped',
                    vscode.TreeItemCollapsibleState.None,
                    this.isRunning ? 'check' : 'error'
                ),
                new ServerStatusItem(
                    'Health',
                    this.isHealthy ? 'Healthy' : 'Not responding',
                    vscode.TreeItemCollapsibleState.None,
                    this.isHealthy ? 'check' : 'warning'
                ),
                new ServerStatusItem(
                    'URL',
                    vscode.workspace.getConfiguration('predic').get('serverUrl', 'http://localhost:8000'),
                    vscode.TreeItemCollapsibleState.None,
                    'globe'
                )
            ];
        }
        return [];
    }
}

class ServerStatusItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly description: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly icon: string
    ) {
        super(label, collapsibleState);
        this.tooltip = `${this.label}: ${this.description}`;
        this.iconPath = new vscode.ThemeIcon(icon);
    }
}