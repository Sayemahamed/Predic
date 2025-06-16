import * as vscode from 'vscode';
import * as path from 'path';
import { fork, ChildProcess } from 'child_process';

// --- FIX: Define the shape of messages for type safety ---
type AgentMessage =
  | { type: 'ready' }
  | { type: 'error', data: { message: string } }
  | { type: 'completionResult', data: { suggestion: string } };
// --- END FIX ---

let agent: ChildProcess | null = null;

export function activate(context: vscode.ExtensionContext) {
    const agentPath = path.join(context.extensionPath, 'dist', 'agent.js');
    agent = fork(agentPath, [], { stdio: 'pipe' });

    agent.on('exit', (code) => {
        console.error(`Predic Agent exited with code: ${code}`);
        agent = null;
    });
    
    agent.stdout?.on('data', (data) => console.log(`[Agent STDOUT]: ${data}`));
    agent.stderr?.on('data', (data) => console.error(`[Agent STDERR]: ${data}`));

    const modelPath = path.join(context.extensionPath, 'dist', 'model');
    agent.send({ type: 'init', data: { modelPath } });
    
    const provider: vscode.InlineCompletionItemProvider = {
        provideInlineCompletionItems(document, position, context, token) {
            return new Promise((resolve) => {
                if (!agent) return resolve([]);

                const textBeforeCursor = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
                const prompt = textBeforeCursor.slice(-2048);
                if (prompt.trim().length === 0) return resolve([]);

                // Use the new AgentMessage type here
                const listener = (message: AgentMessage) => {
                    if (message.type === 'completionResult') {
                        agent?.removeListener('message', listener);
                        if (token.isCancellationRequested) return resolve([]);
                        // The 'data' property is now correctly typed
                        resolve([new vscode.InlineCompletionItem(message.data.suggestion)]);
                    }
                };
                agent.on('message', listener);
                
                token.onCancellationRequested(() => {
                    agent?.removeListener('message', listener);
                    resolve([]);
                });

                agent.send({ type: 'getCompletion', data: { prompt } });
            });
        }
    };

    vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);
    
    // Use the new AgentMessage type here as well
    agent.on('message', (message: AgentMessage) => {
        if (message.type === 'ready') {
            const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
            statusBarItem.text = "$(zap) Predic: Ready";
            statusBarItem.show();
            context.subscriptions.push(statusBarItem);
        } else if (message.type === 'error') {
            // The 'data' property is now correctly typed
            vscode.window.showErrorMessage(`Predic Agent Error: ${message.data.message}`);
        }
    });
}

export function deactivate(): Thenable<void> | undefined {
    if (agent) {
        agent.kill();
    }
    return undefined;
}