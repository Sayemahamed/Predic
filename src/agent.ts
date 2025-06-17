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
  | { type: 'getCompletion'; data: { prompt: string; requestId: string } };

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
        await this.handleCompletionRequest(message.data.prompt, message.data.requestId);
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
    let modelId = 'Xenova/gpt2';
    if (this.checkLocalModel(modelDir)) modelId = modelDir;

    const loadingPromise = pipeline('text-generation', modelId, {
      quantized: true,
      progress_callback: (progress: any) => {
        if (progress.status === 'downloading') {
          const percent = Math.round(progress.progress || 0);
          console.log(`Downloading ${progress.file}: ${percent}%`);
        } else if (progress.status === 'done') {
          console.log(`Downloaded: ${progress.file}`);
        }
      },
    });

    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Model loading timeout (60s)')), 60000)
    );

    this.completionPipeline = (await Promise.race([
      loadingPromise,
      timeoutPromise,
    ])) as TextGenerationPipeline;
  }

  private checkLocalModel(modelDir: string): boolean {
    if (!fs.existsSync(modelDir)) return false;
    const requiredFiles = ['config.json', 'tokenizer_config.json', 'tokenizer.json'];
    return requiredFiles.every(file => fs.readdirSync(modelDir).includes(file));
  }

  private async handleCompletionRequest(prompt: string, requestId: string): Promise<void> {
    if (!this.isInitialized || !this.completionPipeline) {
      return this.sendMessage({ type: 'completionResult', data: { suggestion: '', requestId } });
    }

    const suggestion = await this.generateCompletion(prompt);
    this.sendMessage({ type: 'completionResult', data: { suggestion, requestId } });
  }

  private async generateCompletion(prompt: string): Promise<string> {
    const codePrompt = this.createCodePrompt(prompt);

    const output = await this.completionPipeline!(codePrompt, {
      max_new_tokens: 30,
      temperature: 0.2,
      do_sample: true,
      repetition_penalty: 1.1,
      pad_token_id: 50256,
      eos_token_id: 50256,
      max_time: 5.0,
    });

    return this.extractSuggestion(output, codePrompt);
  }

  private createCodePrompt(prompt: string): string {
    return prompt.split('\n').slice(-3).join('\n');
  }

  private extractSuggestion(output: TextGenerationOutput[], originalPrompt: string): string {
    if (!output?.length) return '';
    let generated = output[0].generated_text;
    if (generated.startsWith(originalPrompt)) {
      generated = generated.slice(originalPrompt.length);
    }
    return generated.trim().split('\n')[0]?.trim().slice(0, 80) || '';
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
