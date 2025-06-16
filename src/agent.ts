// This is the AI Agent. It runs in a separate process.
// It does NOT use any 'vscode' imports.

import path from 'path';
import fs from 'fs';
import os from 'os';
import { env, pipeline } from '@xenova/transformers';

// Define types for clarity
interface TextGenerationOutput {
    generated_text: string;
}
type TextGenerationPipeline = (
    input: string | { role: string; content: string }[],
    options: any
) => Promise<TextGenerationOutput | TextGenerationOutput[]>;

let completionPipeline: TextGenerationPipeline | null = null;

async function initializeModel(modelDir: string) {
    try {
        // Use a reliable path in the user's home directory for the cache
        const cachePath = path.join(os.homedir(), '.predic', 'cache');
        if (!fs.existsSync(cachePath)) {
            fs.mkdirSync(cachePath, { recursive: true });
        }
        
        // Set environment for transformers.js
        process.env.TRANSFORMERS_CACHE = cachePath;
        env.cacheDir = cachePath;
        env.allowRemoteModels = false;

        if (!fs.existsSync(modelDir)) {
            throw new Error(`Agent Error: Model directory not found at ${modelDir}`);
        }

        // Load the pipeline
        completionPipeline = await pipeline('text-generation', modelDir) as TextGenerationPipeline;
        
        // Send a message to the main extension to confirm readiness
        process.send?.({ type: 'ready' });

    } catch (error: any) {
        process.send?.({ type: 'error', data: error.message });
    }
}

async function getCompletion(prompt: string): Promise<string> {
    if (!completionPipeline) {
        return '';
    }

    const messages = [
        { role: "system", content: "You are a helpful code completion assistant for React and Tailwind." },
        { role: "user", content: prompt },
    ];
    
    const output = await completionPipeline(messages, { max_new_tokens: 64 });

    // Simplified and robust output parsing
    let generatedText = Array.isArray(output) ? output[0]?.generated_text : output.generated_text;
    if (!generatedText) return '';

    const assistantMarker = "<|assistant|>";
    const suggestionIndex = generatedText.lastIndexOf(assistantMarker);
    return suggestionIndex !== -1 
        ? generatedText.substring(suggestionIndex + assistantMarker.length).trim() 
        : '';
}


// Listen for messages from the main extension process
process.on('message', async (message: { type: string, data: any }) => {
    switch (message.type) {
        case 'init':
            // The main extension tells the agent where the model is located
            await initializeModel(message.data.modelPath);
            break;
        case 'getCompletion':
            const suggestion = await getCompletion(message.data.prompt);
            // Send the result back to the main extension
            process.send?.({ type: 'completionResult', data: { suggestion } });
            break;
    }
});