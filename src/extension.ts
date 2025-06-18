import * as vscode from 'vscode';
import * as path from 'path';
import { fork, ChildProcess } from 'child_process';

type AgentMessage =
  | { type: 'ready' }
  | { type: 'error'; data: { message: string } }
  | { type: 'completionResult'; data: { suggestion: string; requestId: string } };

type ExtensionMessage =
  | { type: 'init'; data: { modelPath: string } }
  | { type: 'getCompletion'; data: { prompt: string; requestId: string; language: string } };

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
    this.updateStatusBar('$(loading~spin) Predic: Loading SmolLM2...');

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

      // Fork the agent process with more memory for the larger model
      this.agent = fork(agentPath, [], {
        stdio: ['pipe', 'pipe', 'pipe', 'ipc'],
        silent: false,
        cwd: this.context.extensionPath,
        env: {
          ...process.env,
          NODE_ENV: 'production',
          NODE_OPTIONS: '--max-old-space-size=2048' // Increase memory limit
        },
        execArgv: ['--max-old-space-size=2048'] // Also set for the child process
      });

      this.setupAgentListeners();

      // Initialize the model
      const modelPath = path.join(this.context.extensionPath, 'models');
      this.sendMessage({ type: 'init', data: { modelPath } });

      // Wait for ready signal with longer timeout for model download
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
      // Longer timeout for initial model download (5 minutes)
      const timeout = setTimeout(() => {
        console.log('Agent initialization timeout');
        this.updateStatusBar('$(error) Predic: Initialization Timeout');
        resolve(false);
      }, 300000); // 5 minute timeout for model loading

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
        this.updateStatusBar('$(zap) Predic: SmolLM2 Ready');
        vscode.window.showInformationMessage('Predic with SmolLM2-360M-Instruct is ready for intelligent code completion!');
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
      this.statusBarItem.tooltip = 'Predic AI Code Completion with SmolLM2-360M-Instruct';
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

  async getCompletion(prompt: string, language: string): Promise<string> {
    if (!this.agent || !this.isReady) {
      return '';
    }

    const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

    return new Promise<string>((resolve) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        resolve('');
      }, 10000); // 10 second timeout for SmolLM2

      this.pendingRequests.set(requestId, (suggestion: string) => {
        clearTimeout(timeout);
        resolve(suggestion);
      });

      this.sendMessage({
        type: 'getCompletion',
        data: { prompt, requestId, language }
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

// Helper method to clean suggestions
function cleanSuggestion(suggestion: string, currentPrefix: string): string {
  // Remove any part of the suggestion that repeats the current prefix
  const lastWord = currentPrefix.split(/\s+/).pop() || '';
  if (lastWord && suggestion.toLowerCase().startsWith(lastWord.toLowerCase())) {
    suggestion = suggestion.substring(lastWord.length);
  }

  // Remove leading/trailing whitespace but preserve internal structure
  suggestion = suggestion.trim();

  // Don't suggest if it's just whitespace or very short
  if (suggestion.length < 2) {
    return '';
  }

  return suggestion;
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  console.log('Predic extension with SmolLM2-360M-Instruct is being activated...');

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

      // Get the current line to check if we should trigger completion
      const currentLine = document.lineAt(position).text;
      const currentPrefix = currentLine.substring(0, position.character);

      // Skip if at the very beginning of a line with only whitespace
      if (currentPrefix.trim().length === 0 && position.character < 2) {
        return [];
      }

      // Get text before cursor (limited to avoid huge prompts)
      const startLine = Math.max(0, position.line - 10);
      const textBeforeCursor = document.getText(
        new vscode.Range(new vscode.Position(startLine, 0), position)
      );

      // Limit prompt size and clean it
      const prompt = textBeforeCursor.slice(-1000).trim();
      const language = document.languageId;

      console.log(`[Extension] Sending prompt to SmolLM2 agent for ${language}: "${prompt}"`);

      if (prompt.length < 10) {
        return [];
      }

      try {
        const suggestionPromise = predicAgent.getCompletion(prompt, language);
        const timeoutPromise = new Promise<string>(resolve => {
          const timeout = setTimeout(() => resolve(''), 8000);
          token.onCancellationRequested(() => {
            clearTimeout(timeout);
            resolve('');
          });
        });

        const suggestion = await Promise.race([suggestionPromise, timeoutPromise]);

        if (token.isCancellationRequested || !suggestion || suggestion.trim().length === 0) {
          return [];
        }

        // Ensure the suggestion doesn't repeat what's already typed
        const trimmedSuggestion = cleanSuggestion(suggestion, currentPrefix); // Call the standalone function

        if (!trimmedSuggestion) {
          return [];
        }

        // Create the completion item
        const item = new vscode.InlineCompletionItem(trimmedSuggestion);
        item.range = new vscode.Range(position, position);

        console.log(`[Extension] Providing suggestion: "${trimmedSuggestion}"`);
        return [item];
      } catch (error) {
        console.error('Error getting completion:', error);
        return [];
      }
    }
  };

  // Register for more file types including popular programming languages
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
    vscode.window.showInformationMessage('Restarting Predic agent...');
    predicAgent.dispose();
    const newAgent = new PredicAgent(context);
    const success = await newAgent.initialize();
    if (success) {
      vscode.window.showInformationMessage('Predic agent with SmolLM2 restarted successfully!');
    } else {
      vscode.window.showErrorMessage('Failed to restart Predic agent');
    }
  });

  context.subscriptions.push(restartCommand);

  console.log('Predic extension with SmolLM2-360M-Instruct activated successfully!');
}

export function deactivate(): Thenable<void> | undefined {
  console.log('Predic extension is being deactivated...');
  return undefined;
}