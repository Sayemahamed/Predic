import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { pipeline, env } from '@xenova/transformers';

interface TextGenerationOutput {
  generated_text: string;
}

type TextGenerationPipeline = (
  input: string,
  options?: any
) => Promise<TextGenerationOutput[]>;

type AgentMessage =
  | { type: 'ready' }
  | { type: 'error'; data: { message: string } }
  | { type: 'completionResult'; data: { suggestion: string; requestId: string } };

type ExtensionMessage =
  | { type: 'init'; data: { modelPath: string } }
  | { type: 'getCompletion'; data: { prompt: string; requestId: string; language: string } };

class PredicAgent {
  private completionPipeline: TextGenerationPipeline | null = null;
  private isInitialized = false;
  private isInitializing = false;

  constructor() {
    this.setupProcessHandlers();
  }

  private setupProcessHandlers(): void {
    process.on('message', async (message: ExtensionMessage) => {
      try {
        await this.handleMessage(message);
      } catch (error) {
        this.sendMessage({
          type: 'error',
          data: { message: `Failed to handle message: ${error instanceof Error ? error.message : String(error)}` },
        });
      }
    });

    process.on('uncaughtException', (error) => {
      this.sendMessage({
        type: 'error',
        data: { message: `Uncaught exception: ${error.message}` },
      });
    });

    process.on('unhandledRejection', (reason) => {
      this.sendMessage({
        type: 'error',
        data: { message: `Unhandled rejection: ${reason instanceof Error ? reason.message : String(reason)}` },
      });
    });
  }

  private async handleMessage(message: ExtensionMessage): Promise<void> {
    switch (message.type) {
      case 'init':
        await this.initializeModel(message.data.modelPath);
        break;
      case 'getCompletion':
        await this.handleCompletionRequest(
          message.data.prompt, 
          message.data.requestId,
          message.data.language
        );
        break;
    }
  }

  private async initializeModel(modelDir: string): Promise<void> {
    if (this.isInitializing || this.isInitialized) return;
    this.isInitializing = true;

    try {
      const cachePath = path.join(os.homedir(), '.predic', 'cache');
      if (!fs.existsSync(cachePath)) {
        fs.mkdirSync(cachePath, { recursive: true });
      }

      env.cacheDir = cachePath;
      env.allowRemoteModels = true;
      env.allowLocalModels = true;
      env.backends.onnx.logLevel = 'error';

      await this.loadModel(modelDir);
      this.isInitialized = true;
      this.sendMessage({ type: 'ready' });
    } catch (error) {
      this.sendMessage({
        type: 'error',
        data: { message: `Model init failed: ${error instanceof Error ? error.message : String(error)}` },
      });
    } finally {
      this.isInitializing = false;
    }
  }

  private async loadModel(modelDir: string): Promise<void> {
    // List of models to try in order of preference
    const modelCandidates = [
      // 'microsoft/DialoGPT-small',
      // 'Xenova/gpt2',
      // 'Xenova/distilgpt2',
      'HuggingFaceTB/SmolLM2-360M-Instruct'
    ];

    let modelLoaded = false;
    let lastError: Error | null = null;

    // Check if local model exists first
    if (this.checkLocalModel(modelDir)) {
      try {
        console.log(`Loading local model: ${modelDir}`);
        await this.loadModelWithRetry(modelDir);
        modelLoaded = true;
      } catch (error) {
        console.log(`Local model failed, trying remote models: ${error}`);
        lastError = error instanceof Error ? error : new Error(String(error));
      }
    }

    // If local model didn't work, try remote models
    if (!modelLoaded) {
      for (const modelId of modelCandidates) {
        try {
          console.log(`Attempting to load model: ${modelId}`);
          await this.loadModelWithRetry(modelId);
          modelLoaded = true;
          console.log(`Successfully loaded model: ${modelId}`);
          break;
        } catch (error) {
          console.log(`Failed to load ${modelId}: ${error}`);
          lastError = error instanceof Error ? error : new Error(String(error));
        }
      }
    }

    if (!modelLoaded) {
      throw new Error(`Failed to load any model. Last error: ${lastError?.message || 'Unknown error'}`);
    }
  }

  private async loadModelWithRetry(modelId: string, maxRetries: number = 2): Promise<void> {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const loadingPromise = pipeline('text-generation', modelId, {
          quantized: false, // Try without quantization first
          progress_callback: (progress: any) => {
            if (progress.status === 'downloading') {
              const percent = Math.round(progress.progress || 0);
              console.log(`Downloading ${progress.file}: ${percent}%`);
            } else if (progress.status === 'done') {
              console.log(`Downloaded: ${progress.file}`);
            }
          },
        });

        // Increase timeout for model download
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error(`Model loading timeout (180s) - attempt ${attempt}`)), 180000)
        );

        this.completionPipeline = (await Promise.race([
          loadingPromise,
          timeoutPromise,
        ])) as TextGenerationPipeline;

        return; // Success
      } catch (error) {
        console.log(`Attempt ${attempt} failed for ${modelId}: ${error}`);
        
        if (attempt === maxRetries) {
          throw error;
        }
        
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
  }

  private checkLocalModel(modelDir: string): boolean {
    if (!fs.existsSync(modelDir)) return false;
    const requiredFiles = ['config.json'];
    return requiredFiles.some(file => {
      const files = fs.readdirSync(modelDir);
      return files.includes(file);
    });
  }

  private async handleCompletionRequest(prompt: string, requestId: string, language: string): Promise<void> {
    if (!this.isInitialized || !this.completionPipeline) {
      return this.sendMessage({ type: 'completionResult', data: { suggestion: '', requestId } });
    }

    console.log(`[Agent] Received prompt for ${language}: "${prompt}"`);

    const suggestion = await this.generateCompletion(prompt, language);
    this.sendMessage({ type: 'completionResult', data: { suggestion, requestId } });
  }

  private async generateCompletion(prompt: string, language: string): Promise<string> {
    const instructPrompt = this.createInstructPrompt(prompt, language);

    console.log(`[Agent] Instruction prompt: "${instructPrompt}"`);

    try {
      const output = await this.completionPipeline!(instructPrompt, {
        max_new_tokens: 30,
        temperature: 0.3,
        do_sample: true,
        repetition_penalty: 1.1,
        top_p: 0.8,
        top_k: 50,
        pad_token_id: 0,
        max_time: 5.0,
        return_full_text: false,
      });

      console.log(`[Agent] Raw model output: ${JSON.stringify(output)}`);

      return this.extractSuggestion(output, instructPrompt);
    } catch (error) {
      console.error(`[Agent] Generation error: ${error}`);
      return '';
    }
  }

  private createInstructPrompt(prompt: string, language: string): string {
    // Get the last few lines of context
    const lines = prompt.split('\n');
    const contextLines = lines.slice(-3).join('\n');
    
    // For simpler models, use a more direct approach
    return contextLines;
  }

  private extractSuggestion(output: TextGenerationOutput[], originalPrompt: string): string {
    if (!output?.length || !output[0].generated_text) {
      return '';
    }

    let generatedText = output[0].generated_text.trim();
    
    // Take only the first line of the completion
    const firstLine = generatedText.split('\n')[0]?.trim() || '';
    
    // Clean up the suggestion
    let suggestion = firstLine;
    
    // Remove any markdown code blocks
    const codeBlockMatch = suggestion.match(/```(?:\w+)?\s*(.*?)```/s);
    if (codeBlockMatch) {
      suggestion = codeBlockMatch[1].trim();
    }
    
    // Limit length to prevent overly long completions
    suggestion = suggestion.slice(0, 80);
    
    // Remove trailing incomplete tokens
    suggestion = this.cleanupSuggestion(suggestion);
    
    // Don't suggest if it's too short or just whitespace
    if (suggestion.length < 2 || !suggestion.trim()) {
      return '';
    }
    
    console.log(`[Agent] Final suggestion: "${suggestion}"`);
    return suggestion;
  }

  private cleanupSuggestion(suggestion: string): string {
    // Remove incomplete string literals
    if ((suggestion.match(/"/g) || []).length % 2 !== 0) {
      const lastQuote = suggestion.lastIndexOf('"');
      if (lastQuote > 0) {
        suggestion = suggestion.substring(0, lastQuote);
      }
    }
    
    if ((suggestion.match(/'/g) || []).length % 2 !== 0) {
      const lastQuote = suggestion.lastIndexOf("'");
      if (lastQuote > 0) {
        suggestion = suggestion.substring(0, lastQuote);
      }
    }
    
    // Remove incomplete parentheses, brackets, braces
    const openChars = ['(', '[', '{'];
    const closeChars = [')', ']', '}'];
    
    for (let i = 0; i < openChars.length; i++) {
      const openCount = (suggestion.match(new RegExp('\\' + openChars[i], 'g')) || []).length;
      const closeCount = (suggestion.match(new RegExp('\\' + closeChars[i], 'g')) || []).length;
      
      if (openCount > closeCount) {
        const lastOpen = suggestion.lastIndexOf(openChars[i]);
        if (lastOpen > 0) {
          suggestion = suggestion.substring(0, lastOpen);
        }
      }
    }
    
    return suggestion.trim();
  }

  private sendMessage(message: AgentMessage): void {
    if (process.send) {
      process.send(message);
    }
  }
}

new PredicAgent();
process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));