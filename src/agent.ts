import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

// Import transformers directly - it will be externalized by webpack
import { pipeline, env } from '@xenova/transformers';

interface TextGenerationOutput {
  generated_text: string;
}

type TextGenerationPipeline = (
  input: string | { role: string; content: string }[],
  options?: any
) => Promise<TextGenerationOutput | TextGenerationOutput[]>;

type AgentMessage =
  | { type: 'ready' }
  | { type: 'error'; data: { message: string } }
  | { type: 'completionResult'; data: { suggestion: string; requestId: string } };

type ExtensionMessage =
  | { type: 'init'; data: { modelPath: string } }
  | { type: 'getCompletion'; data: { prompt: string; requestId: string } };

class PredicAgent {
  private completionPipeline: TextGenerationPipeline | null = null;
  private isInitialized: boolean = false;

  constructor() {
    this.setupProcessHandlers();
  }

  private setupProcessHandlers(): void {
    process.on('message', async (message: ExtensionMessage) => {
      try {
        await this.handleMessage(message);
      } catch (error) {
        console.error('Error handling message:', error);
        this.sendMessage({
          type: 'error',
          data: { message: `Failed to handle message: ${error}` }
        });
      }
    });

    process.on('uncaughtException', (error) => {
      console.error('Uncaught exception in agent:', error);
      this.sendMessage({
        type: 'error',
        data: { message: `Uncaught exception: ${error.message}` }
      });
    });

    process.on('unhandledRejection', (reason, promise) => {
      console.error('Unhandled rejection in agent:', reason);
      this.sendMessage({
        type: 'error',
        data: { message: `Unhandled rejection: ${reason}` }
      });
    });
  }

  private async handleMessage(message: ExtensionMessage): Promise<void> {
    switch (message.type) {
      case 'init':
        await this.initializeModel(message.data.modelPath);
        break;
      case 'getCompletion':
        await this.handleCompletionRequest(message.data.prompt, message.data.requestId);
        break;
      default:
        console.warn('Unknown message type:', (message as any).type);
    }
  }

  private async initializeModel(modelDir: string): Promise<void> {
    try {
      console.log('Initializing Predic agent...');

      // Setup cache directory
      const cachePath = path.join(os.homedir(), '.predic', 'cache');
      if (!fs.existsSync(cachePath)) {
        fs.mkdirSync(cachePath, { recursive: true });
      }

      // Configure transformers environment
      process.env.TRANSFORMERS_CACHE = cachePath;
      env.cacheDir = cachePath;
      env.allowRemoteModels = true; // Allow downloading if model not found locally

      // Try to load model
      await this.loadModel(modelDir);

      this.isInitialized = true;
      this.sendMessage({ type: 'ready' });
      console.log('Predic agent initialized successfully');

    } catch (error) {
      console.error('Failed to initialize model:', error);
      this.sendMessage({
        type: 'error',
        data: { message: `Model initialization failed: ${error}` }
      });
    }
  }

  private async loadModel(modelDir: string): Promise<void> {
    try {
      // Try local model first, then fallback to a small online model
      let modelId = modelDir;
      
      if (!fs.existsSync(modelDir)) {
        console.log(`Local model not found at ${modelDir}, using fallback model`);
        // Use a small, fast model for development
        modelId = 'microsoft/DialoGPT-small';
      }

      console.log(`Loading model: ${modelId}`);
      
      this.completionPipeline = await pipeline('text-generation', modelId, {
        // Optimize for speed and memory usage
        device: 'cpu',
        dtype: 'fp32',
      }) as TextGenerationPipeline;

      console.log('Model loaded successfully');
    } catch (error) {
      throw new Error(`Model loading failed: ${error}`);
    }
  }

  private async handleCompletionRequest(prompt: string, requestId: string): Promise<void> {
    try {
      if (!this.isInitialized || !this.completionPipeline) {
        this.sendMessage({
          type: 'completionResult',
          data: { suggestion: '', requestId }
        });
        return;
      }

      const suggestion = await this.generateCompletion(prompt);
      
      this.sendMessage({
        type: 'completionResult',
        data: { suggestion, requestId }
      });

    } catch (error) {
      console.error('Error generating completion:', error);
      this.sendMessage({
        type: 'completionResult',
        data: { suggestion: '', requestId }
      });
    }
  }

  private async generateCompletion(prompt: string): Promise<string> {
    if (!this.completionPipeline) {
      return '';
    }

    try {
      // Create a focused prompt for code completion
      const codePrompt = this.createCodePrompt(prompt);
      
      const output = await this.completionPipeline(codePrompt, {
        max_new_tokens: 50,
        temperature: 0.1,
        do_sample: true,
        pad_token_id: 50256, // GPT-2 pad token
        eos_token_id: 50256,
        repetition_penalty: 1.1,
      });

      return this.extractSuggestion(output, prompt);
    } catch (error) {
      console.error('Error in generateCompletion:', error);
      return '';
    }
  }

  private createCodePrompt(prompt: string): string {
    // Extract the last few lines for context
    const lines = prompt.split('\n');
    const contextLines = lines.slice(-5); // Last 5 lines
    
    // Create a simple completion prompt
    return `// Complete this code:\n${contextLines.join('\n')}`;
  }

  private extractSuggestion(output: TextGenerationOutput | TextGenerationOutput[], originalPrompt: string): string {
    try {
      let generatedText = Array.isArray(output) ? output[0]?.generated_text : output.generated_text;
      
      if (!generatedText) {
        return '';
      }

      // Remove the original prompt from the generated text
      const promptStart = generatedText.indexOf(originalPrompt);
      if (promptStart !== -1) {
        generatedText = generatedText.substring(promptStart + originalPrompt.length);
      }

      // Clean up the suggestion
      let suggestion = generatedText.trim();
      
      // Remove common unwanted prefixes/suffixes
      suggestion = suggestion.replace(/^(\/\/.*\n|\/\*.*\*\/\n?)/g, ''); // Remove comments
      suggestion = suggestion.split('\n')[0]; // Take only the first line
      
      // Limit length
      if (suggestion.length > 100) {
        suggestion = suggestion.substring(0, 100);
      }

      return suggestion.trim();
    } catch (error) {
      console.error('Error extracting suggestion:', error);
      return '';
    }
  }

  private sendMessage(message: AgentMessage): void {
    if (process.send) {
      process.send(message);
    }
  }
}

// Initialize the agent
const agent = new PredicAgent();

// Keep the process alive
process.on('SIGTERM', () => {
  console.log('Agent received SIGTERM, shutting down...');
  process.exit(0);
});

console.log('Predic agent process started');