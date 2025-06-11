import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { pipeline, TextGenerationPipeline } from '@xenova/transformers';

// --- Start of Model Handling Logic ---

let pipelineInstance: Promise<TextGenerationPipeline> | null = null;

/**
 * Initializes and retrieves the singleton instance of the text generation pipeline.
 * @param cachePath The absolute path to the directory where models should be stored.
 * @param progress_callback A function to report model loading progress.
 */
async function getPipeline(cachePath: string, progress_callback?: Function): Promise<TextGenerationPipeline> {
    if (pipelineInstance === null) {
        const { env } = await import('@xenova/transformers');
        const modelsDir = path.join(cachePath, 'models');
        if (!fs.existsSync(modelsDir)) {
            fs.mkdirSync(modelsDir, { recursive: true });
        }
        env.cacheDir = modelsDir;
        console.log(`Transformers.js cache path is set to: ${env.cacheDir}`);
        pipelineInstance = pipeline('text-generation', 'onnx-community/gemma-3-1b-it-ONNX-GQA', {
            progress_callback,
        });
    }
    return pipelineInstance;
}

/**
 * Runs the Gemma model to get a code completion suggestion.
 * @param generator The initialized pipeline instance.
 * @param prompt The code context from the editor.
 */
async function runGemma(generator: TextGenerationPipeline, prompt: string): Promise<string> {
    const messages = [
        { role: "system", content: "You are a helpful code completion assistant. Complete the user's code. Provide only the code completion, without any explanation or repeating the user's code." },
        { role: "user", content: prompt },
    ];
    const output = await generator(messages, {
        max_new_tokens: 64,
        do_sample: true,
        temperature: 0.7,
        top_p: 0.9,
    });
    const lastMessage = (output[0] as any).generated_text.at(-1);
    return lastMessage ? lastMessage.content.trim() : '';
}

// --- End of Model Handling Logic ---


export async function activate(context: vscode.ExtensionContext) {
    console.log('Predic is now active in OFFLINE mode!');
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
    statusBarItem.text = "$(loading~spin) Predic: Initializing...";
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    const storagePath = context.globalStoragePath;

    console.log(`[DEBUG] storagePath value: ${storagePath}, type: ${typeof storagePath}`);

    if (typeof storagePath !== 'string' || storagePath.length === 0) {
        const errorMessage = "Could not determine the extension's storage path. Predic cannot start.";
        vscode.window.showErrorMessage(errorMessage);
        console.error(errorMessage);
        statusBarItem.text = "$(error) Predic: Failed";
        return;
    }

    let completionPipeline: TextGenerationPipeline;

    try {
        statusBarItem.text = "$(loading~spin) Predic: Loading AI model...";
        completionPipeline = await getPipeline(storagePath, (progress: any) => {
            console.log(progress);
        });
        statusBarItem.text = "$(zap) Predic: Ready";
        console.log("Offline model is ready.");
    } catch (error: any) {
        statusBarItem.text = "$(error) Predic: Model Failed";
        console.error("Failed to initialize the Predic model:", error);
        vscode.window.showErrorMessage(`Predic failed to load the AI model: ${error.message}`);
        return;
    }

    // --- Debouncing Logic for Performance ---
    let debounceTimer: NodeJS.Timeout | undefined;

    const provider: vscode.InlineCompletionItemProvider = {
        provideInlineCompletionItems: (document, position, context, token) => {
            // Use a promise to handle the debounced result
            return new Promise((resolve) => {
                if (debounceTimer) {
                    clearTimeout(debounceTimer);
                }

                debounceTimer = setTimeout(async () => {
                    const textBeforeCursor = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
                    if (textBeforeCursor.trim().length < 10) {
                        return resolve([]); // Resolve with no items if not enough text
                    }

                    try {
                        console.log("Requesting completion...");
                        const suggestion = await runGemma(completionPipeline, textBeforeCursor);
                        if (token.isCancellationRequested || !suggestion) {
                            return resolve([]);
                        }
                        // Resolve with the completion item
                        resolve([new vscode.InlineCompletionItem(suggestion)]);
                    } catch (error: any) {
                        console.error("Error during debounced inference:", error);
                        resolve([]); // Resolve with no items on error
                    }
                }, 300); // Wait for 300ms of inactivity before triggering
            });
        },
    };

    vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);
}

export function deactivate() {}
