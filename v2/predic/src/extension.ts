import * as vscode from 'vscode';
import { ServerManager } from './serverManager';
import { ChatViewProvider } from './providers/chatViewProvider';
import { PredicInlineCompletionProvider } from './providers/inlineCompletionProvider';
import { ModelManagerProvider } from './providers/modelManagerProvider';

export async function activate(context: vscode.ExtensionContext) {
    const serverManager = new ServerManager(context);
    
    // Providers
    const chatProvider = new ChatViewProvider(context.extensionUri);
    const modelManagerProvider = new ModelManagerProvider(context.extensionUri, context);
    const inlineProvider = new PredicInlineCompletionProvider();

    // Register View
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider("predic.chatView", chatProvider)
    );

    // Commands
    context.subscriptions.push(
        // Basic
        vscode.commands.registerCommand('predic.openModelManager', () => modelManagerProvider.show()),
        vscode.commands.registerCommand('predic.startServer', () => serverManager.start()),
        vscode.commands.registerCommand('predic.stopServer', () => serverManager.stop()),
        
        // Smart Actions
        vscode.commands.registerCommand('predic.explainCode', () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && !editor.selection.isEmpty) {
                const code = editor.document.getText(editor.selection);
                chatProvider.triggerAnalysis("Explain this code snippet in detail.", code);
            } else {
                vscode.window.showInformationMessage("Please select some code first.");
            }
        }),
        
        vscode.commands.registerCommand('predic.fixCode', () => {
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                const code = editor.selection.isEmpty 
                    ? editor.document.getText() 
                    : editor.document.getText(editor.selection);
                
                chatProvider.triggerAnalysis("Find and fix any errors in this code. Return the fixed code block.", code);
            }
        }),

        vscode.commands.registerCommand('predic.restartServer', async () => {
             // ... existing restart logic ...
             await serverManager.stop();
             await new Promise(r => setTimeout(r, 1000));
             await serverManager.start();
        })
    );

    // Inline Completion
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider({ pattern: "**" }, inlineProvider)
    );

    await serverManager.start();
}

export function deactivate() {}