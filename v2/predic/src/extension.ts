import * as vscode from 'vscode';
import { ServerManager } from './serverManager';
import { ChatViewProvider } from './providers/chatViewProvider';
import { PredicInlineCompletionProvider } from './providers/inlineCompletionProvider';

export async function activate(context: vscode.ExtensionContext) {
    console.log('Predic is now active!');

    // 1. Start the Server
    const serverManager = new ServerManager(context);
    await serverManager.start();

    // 2. Register Chat View
    const chatProvider = new ChatViewProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider("predic.chatView", chatProvider)
    );

    // 3. Register Inline Completion (Ghost Text)
    const inlineProvider = new PredicInlineCompletionProvider();
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider(
            { pattern: "**" }, // Trigger on all files
            inlineProvider
        )
    );

    // Cleanup on deactivate
    context.subscriptions.push({
        dispose: () => serverManager.stop()
    });
}

export function deactivate() {}