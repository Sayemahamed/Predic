import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

export async function activate(context: vscode.ExtensionContext) {
    
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
    context.subscriptions.push(statusBarItem);
    statusBarItem.text = "$(loading~spin) Predic: Starting...";
    statusBarItem.show();

    try {
        console.log("--- Predic Activation Start ---");
        // step1
        const storageUri = context.globalStorageUri;
        console.log(`[Step 1] VS Code provided storageUri:`, storageUri);
        // step2
        const storagePath = storageUri.fsPath;
        console.log(`[Step 2] Converted to fsPath: ${storagePath}, type: ${typeof storagePath}`);

        if (!storagePath || typeof storagePath !== 'string') {
            throw new Error("Could not determine a valid storage path from VS Code's context.");
        }
        // step3
        if (!fs.existsSync(storagePath)) {
            console.log(`[Step 3] Storage path does not exist. Creating directory...`);
            fs.mkdirSync(storagePath, { recursive: true });
        } else {
            console.log(`[Step 3] Storage path already exists.`);
        }
        // step4
        process.env.TRANSFORMERS_CACHE = storagePath;
        console.log(`[Step 4] Set process.env.TRANSFORMERS_CACHE to: ${storagePath}`);
        // step5
        console.log("[Step 5] Dynamically importing '@xenova/transformers'...");
        const { env, pipeline } = await import('@xenova/transformers'); 
        
        // <--- error here the extension stops while compiling this line above

        console.log("[Step 5] Library imported successfully.");
        // step6
        env.cacheDir = storagePath;
        console.log(`[Step 6] Also explicitly set library env.cacheDir to: ${storagePath}`);

        statusBarItem.text = "$(sync~spin) Predic: Loading model...";
        // step7
        const modelPath = path.join(context.extensionPath, 'dist', 'model');
        console.log(`[Step 7] Attempting to load model from: ${modelPath}`);
        
        env.allowRemoteModels = false;

        const completionPipeline: any = await pipeline('text-generation', modelPath);
        // step8
        statusBarItem.text = "$(zap) Predic: Ready";
        console.log("[Step 8] Model pipeline loaded successfully. Predic is ready.");

        async function runGemma(prompt: string): Promise<string> {
            const messages = [
                { role: "system", content: "You are a helpful code completion assistant for React and Tailwind. Complete the user's code." },
                { role: "user", content: prompt },
            ];
            const output = await completionPipeline(messages, {
                max_new_tokens: 64,
                do_sample: true,
                temperature: 0.7,
                top_p: 0.9,
            });
            const lastMessage = (output[0] as any).generated_text.at(-1);
            return lastMessage ? lastMessage.content.trim() : '';
        }

        let debounceTimer: NodeJS.Timeout | undefined;
        const provider: vscode.InlineCompletionItemProvider = {
            provideInlineCompletionItems: (document, position, context, token) => {
                return new Promise((resolve) => {
                    if (debounceTimer) clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(async () => {
                        const textBeforeCursor = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
                        
                        try {
                            console.log(`--- Sending prompt to AI ---\n${textBeforeCursor}`);
                            const suggestion = await runGemma(textBeforeCursor);
                            console.log(`--- Received suggestion: "${suggestion}" ---`);
                            if (token.isCancellationRequested || !suggestion) return resolve([]);
                            resolve([new vscode.InlineCompletionItem(suggestion)]);
                        } catch (error: any) {
                            console.error("Error during inference:", error);
                            resolve([]);
                        }
                    }, 300);
                });
            },
        };

        vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);

    } catch (error: any) {
        statusBarItem.text = "$(error) Predic: Failed";
        console.error("--- PREDIC ACTIVATION FAILED ---", error);
        vscode.window.showErrorMessage(`Predic failed to start. Please check the debug console. Error: ${error.message}`);
    }
}

export function deactivate() {}
