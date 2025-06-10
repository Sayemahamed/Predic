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
        // Dynamically import the 'env' object from the library.
        const { env } = await import('@xenova/transformers');
        
        // Define the models directory within the provided cache path.
        const modelsDir = path.join(cachePath, 'models');
        
        // Ensure the directory exists before setting the cache path.
        if (!fs.existsSync(modelsDir)) {
            fs.mkdirSync(modelsDir, { recursive: true });
        }
        
        // Set the cache directory for transformers.js
        env.cacheDir = modelsDir;
        console.log(`Transformers.js cache path is set to: ${env.cacheDir}`);

        // Create the pipeline promise. This will download the model on the first run.
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

    // Cast the output to 'any' to safely access 'generated_text'.
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

    // FIXED: Switched to the deprecated but more direct `globalStoragePath`
    // to prevent the 'undefined' path error. This is a common workaround for this issue.
    const storagePath = context.globalStoragePath;

    // Added a more explicit check to ensure the path is a valid, non-empty string.
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

    const provider: vscode.InlineCompletionItemProvider = {
        async provideInlineCompletionItems(document, position, context, token) {
            const textBeforeCursor = document.getText(new vscode.Range(new vscode.Position(0, 0), position));

            if (textBeforeCursor.trim().length < 10) return;

            try {
                const suggestion = await runGemma(completionPipeline, textBeforeCursor);
                if (token.isCancellationRequested || !suggestion) return;
                return [new vscode.InlineCompletionItem(suggestion)];
            } catch (error: any) {
                console.error("Error during offline inference:", error);
                return;
            }
        },
    };

    vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);
}

export function deactivate() {}
