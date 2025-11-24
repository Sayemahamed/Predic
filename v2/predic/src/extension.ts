import * as vscode from 'vscode';
import { ServerManager } from './serverManager';
import { ChatViewProvider } from './providers/chatViewProvider';
import { PredicInlineCompletionProvider } from './providers/inlineCompletionProvider';
import { ModelManagerProvider } from './providers/modelManagerProvider';

export async function activate(context: vscode.ExtensionContext) {
    const serverManager = new ServerManager(context);
    
    // 1. Register Providers
    const chatProvider = new ChatViewProvider(context.extensionUri);
    const modelManagerProvider = new ModelManagerProvider(context.extensionUri, context);
    const inlineProvider = new PredicInlineCompletionProvider();

    // 2. Register Views
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider("predic.chatView", chatProvider)
    );

    // 3. Register Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('predic.openModelManager', () => {
            modelManagerProvider.show();
        }),
        vscode.commands.registerCommand('predic.startServer', () => serverManager.start()),
        vscode.commands.registerCommand('predic.stopServer', () => serverManager.stop()),
        vscode.commands.registerCommand('predic.restartServer', async () => {
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Restarting Predic Server...",
                cancellable: false
            }, async () => {
                await serverManager.stop();
                await new Promise(resolve => setTimeout(resolve, 1000)); 
                await serverManager.start();
            });
        })
    );

    // 4. Register Inline Completion
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider(
            { pattern: "**" }, 
            inlineProvider
        )
    );

    // Auto-start
    await serverManager.start();
}

export function deactivate() {}