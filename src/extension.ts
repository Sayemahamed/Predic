import * as vscode from 'vscode';
import * as path from 'path';
import { pipeline, TextGenerationPipeline, env } from '@xenova/transformers';

// This will hold the model pipeline instance once it's created.
let pipelineInstance: Promise<TextGenerationPipeline> | null = null;

// Initializes and retrieves the pipeline, loading the model from a local path.
async function getPipeline(extensionPath: string): Promise<TextGenerationPipeline> {
    if (pipelineInstance === null) {
        // Configure the library to only look for models in a local directory.
        env.allowRemoteModels = false;
        env.localModelPath = path.join(extensionPath, 'src', 'model');

        // Create the pipeline instance from the local model path.
        pipelineInstance = pipeline('text-generation', env.localModelPath);
    }
    return pipelineInstance;
}

// Runs the Gemma model to get a code completion suggestion.
async function runGemma(generator: TextGenerationPipeline, prompt: string): Promise<string> {
    // Format the prompt into the chat structure the model expects.
    const messages = [
        { role: "system", content: "You are a helpful code completion assistant. Complete the user's code. Provide only the code completion, without any explanation or repeating the user's code." },
        { role: "user", content: prompt },
    ];

    // Generate text based on the formatted prompt.
    const output = await generator(messages, {
        max_new_tokens: 64,
        do_sample: true,
        temperature: 0.7,
        top_p: 0.9,
    });

    // Extract and return the model's suggested text.
    const lastMessage = (output[0] as any).generated_text.at(-1);
    return lastMessage ? lastMessage.content.trim() : '';
}

// This function is called when your extension is activated.
export async function activate(context: vscode.ExtensionContext) {
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
    context.subscriptions.push(statusBarItem);

    let completionPipeline: TextGenerationPipeline;

    try {
        // Load the local model and show status updates in the status bar.
        statusBarItem.text = "$(sync~spin) Predic: Loading local model...";
        completionPipeline = await getPipeline(context.extensionPath);
        statusBarItem.text = "$(zap) Predic: Ready";

    } catch (error: any) {
        // Handle any errors during model initialization.
        statusBarItem.text = "$(error) Predic: Failed to load";
        console.error("Failed to initialize the Predic model:", error);
        vscode.window.showErrorMessage(`Predic failed to load the AI model. Error: ${error.message}`);
        return;
    }

    // This object provides the inline completion items to VS Code.
    let debounceTimer: NodeJS.Timeout | undefined;
    const provider: vscode.InlineCompletionItemProvider = {
        provideInlineCompletionItems: (document, position, context, token) => {
            return new Promise((resolve) => {
                // Use a debounce timer to avoid making requests on every keystroke.
                if (debounceTimer) clearTimeout(debounceTimer);

                debounceTimer = setTimeout(async () => {
                    const textBeforeCursor = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
                    if (textBeforeCursor.trim().length < 10) return resolve([]);

                    try {
                        // Get a suggestion from the model and resolve the promise.
                        const suggestion = await runGemma(completionPipeline, textBeforeCursor);
                        if (token.isCancellationRequested || !suggestion) return resolve([]);
                        resolve([new vscode.InlineCompletionItem(suggestion)]);
                    } catch (error: any) {
                        console.error("Error during inference:", error);
                        resolve([]);
                    }
                }, 300); // Wait 300ms after the user stops typing.
            });
        },
    };

    // Register our completion provider for all languages.
    vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);
}

// This function is called when your extension is deactivated.
export function deactivate() {}
