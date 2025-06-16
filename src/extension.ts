import * as vscode from 'vscode';
import * as path from 'path';
import { fork, ChildProcess } from 'child_process';

type AgentMessage =
  | { type: 'ready' }
  | { type: 'error'; data: { message: string } }
  | { type: 'completionResult'; data: { suggestion: string; requestId: string } };

type ExtensionMessage =
  | { type: 'init'; data: { modelPath: string } }
  | { type: 'getCompletion'; data: { prompt: string; requestId: string } };

class PredicAgent {
  private agent: ChildProcess | null = null;
  private isReady: boolean = false;
  private pendingRequests = new Map<string, (suggestion: string) => void>();
  private statusBarItem: vscode.StatusBarItem | null = null;

  constructor(private context: vscode.ExtensionContext) {}

  async initialize(): Promise<boolean> {
    try {
      const agentPath = path.join(this.context.extensionPath, 'dist', 'agent.js');
      
      // Check if agent.js exists
      try {
        await vscode.workspace.fs.stat(vscode.Uri.file(agentPath));
      } catch {
        vscode.window.showErrorMessage('Predic: Agent file not found. Please build the extension first.');
        return false;
      }

      this.agent = fork(agentPath, [], { 
        stdio: 'pipe',
        silent: true 
      });

      this.setupAgentListeners();
      
      // Initialize the model
      const modelPath = path.join(this.context.extensionPath, 'models');
      this.sendMessage({ type: 'init', data: { modelPath } });

      return true;
    } catch (error) {
      console.error('Failed to initialize Predic agent:', error);
      vscode.window.showErrorMessage(`Predic: Failed to initialize agent - ${error}`);
      return false;
    }
  }

  private setupAgentListeners(): void {
    if (!this.agent) return;

    this.agent.on('message', (message: AgentMessage) => {
      this.handleAgentMessage(message);
    });

    this.agent.on('exit', (code, signal) => {
      console.log(`Predic agent exited with code ${code}, signal ${signal}`);
      this.isReady = false;
      this.updateStatusBar('$(error) Predic: Agent Stopped');
      
      // Clear pending requests
      this.pendingRequests.forEach(resolve => resolve(''));
      this.pendingRequests.clear();
    });

    this.agent.on('error', (error) => {
      console.error('Predic agent error:', error);
      vscode.window.showErrorMessage(`Predic Agent Error: ${error.message}`);
    });

    this.agent.stdout?.on('data', (data) => {
      console.log(`[Predic Agent STDOUT]: ${data}`);
    });

    this.agent.stderr?.on('data', (data) => {
      console.error(`[Predic Agent STDERR]: ${data}`);
    });
  }

  private handleAgentMessage(message: AgentMessage): void {
    switch (message.type) {
      case 'ready':
        this.isReady = true;
        this.updateStatusBar('$(zap) Predic: Ready');
        vscode.window.showInformationMessage('Predic is ready for code completion!');
        break;

      case 'error':
        this.isReady = false;
        this.updateStatusBar('$(error) Predic: Error');
        vscode.window.showErrorMessage(`Predic: ${message.data.message}`);
        break;

      case 'completionResult':
        const resolver = this.pendingRequests.get(message.data.requestId);
        if (resolver) {
          resolver(message.data.suggestion);
          this.pendingRequests.delete(message.data.requestId);
        }
        break;
    }
  }

  private updateStatusBar(text: string): void {
    if (!this.statusBarItem) {
      this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
      this.context.subscriptions.push(this.statusBarItem);
    }
    this.statusBarItem.text = text;
    this.statusBarItem.show();
  }

  private sendMessage(message: ExtensionMessage): void {
    if (this.agent) {
      this.agent.send(message);
    }
  }

  async getCompletion(prompt: string): Promise<string> {
    if (!this.agent || !this.isReady) {
      return '';
    }

    const requestId = Math.random().toString(36).substring(7);
    
    return new Promise<string>((resolve) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        resolve('');
      }, 5000); // 5 second timeout

      this.pendingRequests.set(requestId, (suggestion: string) => {
        clearTimeout(timeout);
        resolve(suggestion);
      });

      this.sendMessage({ 
        type: 'getCompletion', 
        data: { prompt, requestId } 
      });
    });
  }

  dispose(): void {
    if (this.agent) {
      this.agent.kill('SIGTERM');
      this.agent = null;
    }
    this.isReady = false;
    this.pendingRequests.clear();
  }
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  console.log('Predic extension is being activated...');

  const predicAgent = new PredicAgent(context);
  
  // Initialize the agent
  const initialized = await predicAgent.initialize();
  if (!initialized) {
    return;
  }

  // Register the completion provider
  const provider: vscode.InlineCompletionItemProvider = {
    async provideInlineCompletionItems(
      document: vscode.TextDocument,
      position: vscode.Position,
      context: vscode.InlineCompletionContext,
      token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[]> {
      
      // Skip if cancellation is already requested
      if (token.isCancellationRequested) {
        return [];
      }

      // Get text before cursor (limited to avoid huge prompts)
      const textBeforeCursor = document.getText(
        new vscode.Range(new vscode.Position(Math.max(0, position.line - 10), 0), position)
      );
      
      const prompt = textBeforeCursor.slice(-1000).trim(); // Last 1000 chars
      
      if (prompt.length === 0) {
        return [];
      }

      try {
        const suggestion = await Promise.race([
          predicAgent.getCompletion(prompt),
          new Promise<string>(resolve => {
            token.onCancellationRequested(() => resolve(''));
            setTimeout(() => resolve(''), 3000); // 3 second timeout
          })
        ]);

        if (token.isCancellationRequested || !suggestion.trim()) {
          return [];
        }

        return [new vscode.InlineCompletionItem(suggestion.trim())];
      } catch (error) {
        console.error('Error getting completion:', error);
        return [];
      }
    }
  };

  // Register for all supported file types
  const selector: vscode.DocumentSelector = [
    { language: 'javascript' },
    { language: 'typescript' },
    { language: 'javascriptreact' },
    { language: 'typescriptreact' },
    { language: 'html' },
    { language: 'css' },
    { language: 'json' }
  ];

  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider(selector, provider),
    { dispose: () => predicAgent.dispose() }
  );

  console.log('Predic extension activated successfully!');
}

export function deactivate(): Thenable<void> | undefined {
  console.log('Predic extension is being deactivated...');
  return undefined;
}