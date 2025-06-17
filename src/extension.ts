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
  private isInitializing: boolean = false;
  private pendingRequests = new Map<string, (suggestion: string) => void>();
  private statusBarItem: vscode.StatusBarItem | null = null;
  private initializationPromise: Promise<boolean> | null = null;

  constructor(private context: vscode.ExtensionContext) {}

  async initialize(): Promise<boolean> {
    if (this.initializationPromise) {
      return this.initializationPromise;
    }

    this.initializationPromise = this.doInitialize();
    return this.initializationPromise;
  }

  private async doInitialize(): Promise<boolean> {
    if (this.isInitializing) {
      return false;
    }

    this.isInitializing = true;
    this.updateStatusBar('$(loading~spin) Predic: Initializing...');

    try {
      const agentPath = path.join(this.context.extensionPath, 'dist', 'agent.js');
      
      // Check if agent.js exists
      try {
        await vscode.workspace.fs.stat(vscode.Uri.file(agentPath));
      } catch {
        const message = 'Predic: Agent file not found. Please build the extension first using "npm run build".';
        vscode.window.showErrorMessage(message);
        this.updateStatusBar('$(error) Predic: Build Required');
        this.isInitializing = false;
        return false;
      }

      // Fork the agent process
      this.agent = fork(agentPath, [], { 
        stdio: ['pipe', 'pipe', 'pipe', 'ipc'],
        silent: false,
        cwd: this.context.extensionPath,
        env: {
          ...process.env,
          NODE_ENV: 'production'
        }
      });

      this.setupAgentListeners();
      
      // Initialize the model
      const modelPath = path.join(this.context.extensionPath, 'models');
      this.sendMessage({ type: 'init', data: { modelPath } });

      // Wait for ready signal with timeout
      const success = await this.waitForReady();
      this.isInitializing = false;
      return success;

    } catch (error) {
      console.error('Failed to initialize Predic agent:', error);
      this.isInitializing = false;
      this.updateStatusBar('$(error) Predic: Failed to Initialize');
      vscode.window.showErrorMessage(`Predic: Failed to initialize agent - ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  private waitForReady(): Promise<boolean> {
    return new Promise<boolean>((resolve) => {
      const timeout = setTimeout(() => {
        console.log('Agent initialization timeout');
        resolve(false);
      }, 120000); // 2 minute timeout for model loading

      const messageHandler = (message: AgentMessage) => {
        if (message.type === 'ready') {
          clearTimeout(timeout);
          this.agent?.off('message', messageHandler);
          resolve(true);
        } else if (message.type === 'error') {
          clearTimeout(timeout);
          this.agent?.off('message', messageHandler);
          resolve(false);
        }
      };

      if (this.agent) {
        this.agent.on('message', messageHandler);
      } else {
        clearTimeout(timeout);
        resolve(false);
      }
    });
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

      // Show error message if unexpected exit
      if (code !== 0 && code !== null) {
        vscode.window.showErrorMessage(`Predic agent crashed with exit code ${code}`);
      }
    });

    this.agent.on('error', (error) => {
      console.error('Predic agent error:', error);
      this.updateStatusBar('$(error) Predic: Agent Error');
      vscode.window.showErrorMessage(`Predic Agent Error: ${error.message}`);
    });

    // Better logging for debugging
    this.agent.stdout?.on('data', (data) => {
      const message = data.toString().trim();
      if (message) {
        console.log(`[Predic Agent]: ${message}`);
      }
    });

    this.agent.stderr?.on('data', (data) => {
      const message = data.toString().trim();
      if (message) {
        console.error(`[Predic Agent Error]: ${message}`);
      }
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
        console.error('Agent error:', message.data.message);
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
      this.statusBarItem.tooltip = 'Predic AI Code Completion';
      this.context.subscriptions.push(this.statusBarItem);
    }
    this.statusBarItem.text = text;
    this.statusBarItem.show();
  }

  private sendMessage(message: ExtensionMessage): void {
    if (this.agent && this.agent.connected) {
      this.agent.send(message);
    } else {
      console.error('Cannot send message: agent not connected');
    }
  }

  async getCompletion(prompt: string): Promise<string> {
    if (!this.agent || !this.isReady) {
      return '';
    }

    const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    
    return new Promise<string>((resolve) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        resolve('');
      }, 8000); // 8 second timeout

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
    this.isInitializing = false;
    this.pendingRequests.clear();
    this.initializationPromise = null;
  }
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  console.log('Predic extension is being activated...');

  const predicAgent = new PredicAgent(context);
  
  // Initialize the agent
  const initialized = await predicAgent.initialize();
  if (!initialized) {
    console.error('Failed to initialize Predic agent');
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

      // Skip for very short files or at the beginning
      if (position.line < 1 || document.lineCount < 2) {
        return [];
      }

      // Get text before cursor (limited to avoid huge prompts)
      const startLine = Math.max(0, position.line - 20);
      const textBeforeCursor = document.getText(
        new vscode.Range(new vscode.Position(startLine, 0), position)
      );
      
      // Limit prompt size and clean it
      const prompt = textBeforeCursor.slice(-800).trim();
      
      if (prompt.length < 10) {
        return [];
      }

      try {
        const suggestionPromise = predicAgent.getCompletion(prompt);
        const timeoutPromise = new Promise<string>(resolve => {
          const timeout = setTimeout(() => resolve(''), 5000);
          token.onCancellationRequested(() => {
            clearTimeout(timeout);
            resolve('');
          });
        });

        const suggestion = await Promise.race([suggestionPromise, timeoutPromise]);

        if (token.isCancellationRequested || !suggestion || suggestion.trim().length === 0) {
          return [];
        }

        // Create the completion item
        const item = new vscode.InlineCompletionItem(suggestion.trim());
        item.range = new vscode.Range(position, position);
        
        return [item];
      } catch (error) {
        console.error('Error getting completion:', error);
        return [];
      }
    }
  };

  // Register for more file types
  const selector: vscode.DocumentSelector = [
    { language: 'javascript' },
    { language: 'typescript' },
    { language: 'javascriptreact' },
    { language: 'typescriptreact' },
    { language: 'css' },
  ];

  const disposable = vscode.languages.registerInlineCompletionItemProvider(selector, provider);
  
  context.subscriptions.push(
    disposable,
    { dispose: () => predicAgent.dispose() }
  );

  // Add a command to restart the agent
  const restartCommand = vscode.commands.registerCommand('predic.restart', async () => {
    predicAgent.dispose();
    const newAgent = new PredicAgent(context);
    const success = await newAgent.initialize();
    if (success) {
      vscode.window.showInformationMessage('Predic agent restarted successfully!');
    } else {
      vscode.window.showErrorMessage('Failed to restart Predic agent');
    }
  });

  context.subscriptions.push(restartCommand);

  console.log('Predic extension activated successfully!');
}

export function deactivate(): Thenable<void> | undefined {
  console.log('Predic extension is being deactivated...');
  return undefined;
}