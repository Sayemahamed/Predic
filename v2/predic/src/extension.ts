import * as vscode from 'vscode';
import { ServerManager } from './serverManager';
import { ChatViewProvider } from './providers/chatViewProvider';
import { PredicInlineCompletionProvider } from './providers/inlineCompletionProvider';
import { ModelManagerProvider } from './providers/modelManagerProvider';

export async function activate(context: vscode.ExtensionContext) {
    const serverManager = new ServerManager(context);
    
    const chatProvider = new ChatViewProvider(context.extensionUri);
    const modelManagerProvider = new ModelManagerProvider(context.extensionUri, context);
    const inlineProvider = new PredicInlineCompletionProvider();

    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider("predic.chatView", chatProvider)
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('predic.openModelManager', () => modelManagerProvider.show()),
        vscode.commands.registerCommand('predic.clearChat', () => chatProvider.clearChat()),
        
        // 1. Close Command
        vscode.commands.registerCommand('predic.closeChat', () => {
            vscode.commands.executeCommand('workbench.action.toggleSidebarVisibility');
        }),

        // 2. Full Screen Command
        vscode.commands.registerCommand('predic.openInEditor', () => {
            chatProvider.openInEditor();
        }),

        vscode.commands.registerCommand('predic.startServer', () => serverManager.start()),
        vscode.commands.registerCommand('predic.stopServer', () => serverManager.stop()),
        vscode.commands.registerCommand('predic.restartServer', async () => {
             await serverManager.stop();
             await new Promise(r => setTimeout(r, 1000));
             await serverManager.start();
        }),

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
        })
    );

    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider({ pattern: "**" }, inlineProvider)
    );

    await serverManager.start();
}

export function deactivate() {}