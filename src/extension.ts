import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os'; // Import the 'os' module to get the home directory

// --- Type Definitions (Unchanged) ---
interface TextGenerationOutput {
    generated_text: string;
}
type TextGenerationPipeline = (
    input: string | { role: string; content: string }[],
    options: any
) => Promise<TextGenerationOutput | TextGenerationOutput[]>;

let predicPipeline: Promise<TextGenerationPipeline | null> | null = null;

export function activate(context: vscode.ExtensionContext) {
    initializePredic(context);

    const provider: vscode.InlineCompletionItemProvider = {
        async provideInlineCompletionItems(document, position, context, token) {
            const pipeline = await predicPipeline;
            if (!pipeline || token.isCancellationRequested) return [];
            
            const textBeforeCursor = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
            const contextWindow = textBeforeCursor.slice(-2048);
            if (contextWindow.trim().length === 0) return [];

            try {
                const suggestion = await getCompletion(pipeline, contextWindow);
                if (!suggestion) return [];
                return [new vscode.InlineCompletionItem(suggestion)];
            } catch (error: any) {
                console.error("Error during inference:", error);
                return [];
            }
        },
    };

    vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);
}

function initializePredic(context: vscode.ExtensionContext) {
    predicPipeline = new Promise((resolve) => {
        // We don't need withProgress here because the agent pattern
        // implies this happens quickly or in the background.
        // For simplicity, we'll still do it in one flow but with better path handling.
        
        (async () => {
            try {
                console.log("[Predic] Starting initialization...");

                // TABBYML-INSPIRED FIX: Use a reliable path instead of globalStorageUri.
                // We will create a .predic folder in the user's home directory.
                const homeDir = os.homedir();
                if (!homeDir) {
                    throw new Error("Could not determine user's home directory.");
                }
                const storagePath = path.join(homeDir, '.predic', 'cache');
                console.log(`[Predic] Using reliable storage path: ${storagePath}`);

                // Create the directory if it doesn't exist.
                if (!fs.existsSync(storagePath)) {
                    fs.mkdirSync(storagePath, { recursive: true });
                }
                
                // Set the cache directory BEFORE importing transformers
                process.env.TRANSFORMERS_CACHE = storagePath;

                const { env, pipeline } = await import('@xenova/transformers');
                
                // Redundantly set it for the library's environment
                env.cacheDir = storagePath;
                env.allowRemoteModels = false;

                const modelPath = path.join(context.extensionPath, 'dist', 'model');
                if (!fs.existsSync(modelPath)) {
                    throw new Error(`Model directory not found at path: ${modelPath}. Make sure the model is copied to 'dist/model' during the build process.`);
                }
                
                console.log("[Predic] Loading model pipeline...");
                const loadedPipeline = await pipeline('text-generation', modelPath) as TextGenerationPipeline;

                const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
                statusBarItem.text = "$(zap) Predic: Ready";
                statusBarItem.show();
                context.subscriptions.push(statusBarItem);
                
                console.log("[Predic] Initialization successful.");
                resolve(loadedPipeline);

            } catch (error: any) {
                console.error("--- PREDIC ACTIVATION FAILED ---", error);
                vscode.window.showErrorMessage(`Predic failed to start. Error: ${error.message}`);
                resolve(null);
            }
        })();
    });
}


// --- getCompletion and deactivate functions (Unchanged) ---
async function getCompletion(pipeline: TextGenerationPipeline, prompt: string): Promise<string> {
    const messages = [
        { role: "system", content: "You are a helpful code completion assistant for React and Tailwind. Complete the user's code snippet." },
        { role: "user", content: prompt },
    ];
    
    const output = await pipeline(messages, {
        max_new_tokens: 64,
        do_sample: true,
        temperature: 0.7,
        top_p: 0.9,
    });

    let generatedText: string | undefined;
    if (Array.isArray(output)) {
        generatedText = output[0]?.generated_text;
    } else {
        generatedText = output.generated_text;
    }

    if (!generatedText) return '';

    const assistantMarker = "<|assistant|>";
    const suggestionStartIndex = generatedText.lastIndexOf(assistantMarker);
    const suggestion = suggestionStartIndex !== -1 
        ? generatedText.substring(suggestionStartIndex + assistantMarker.length).trim()
        : '';
    
    return suggestion;
}

export function deactivate() {
    predicPipeline = null;
}