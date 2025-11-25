import * as vscode from 'vscode';
import axios from 'axios';

export class PredicInlineCompletionProvider implements vscode.InlineCompletionItemProvider {
    
    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[]> {
        
        // 1. Get Config
        const config = vscode.workspace.getConfiguration('predic');
        const port = config.get<number>('port') || 5001;
        const modelName = config.get<string>('modelPath') || "";
        
        // 2. Get Context (Prefix and Suffix)
        const prefix = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
        const suffix = document.getText(new vscode.Range(position, document.positionAt(document.getText().length)));

        // 3. Detect Model Capability (Rough check)
        // Qwen and DeepSeek usually support FIM
        const supportsFim = modelName.toLowerCase().includes('qwen') || modelName.toLowerCase().includes('deepseek');

        let prompt = prefix;
        
        // 4. Apply FIM Format if supported
        if (supportsFim) {
            // Standard FIM format for Qwen/CodeLlama
            prompt = `<|fim_prefix|>${prefix}<|fim_suffix|>${suffix}<|fim_middle|>`;
        }

        try {
            const response = await axios.post(`http://localhost:${port}/v1/completions`, {
                prompt: prompt,
                max_tokens: 64, // Keep it short for ghost text
                stop: ["\n", "<|file_separator|>"], // Stop at newlines usually
                temperature: 0.1,
                top_p: 0.9
            });

            const prediction = response.data.choices[0].text;

            if (!prediction || prediction.trim() === "") return [];

            return [new vscode.InlineCompletionItem(prediction, new vscode.Range(position, position))];

        } catch (err) {
            return [];
        }
    }
}