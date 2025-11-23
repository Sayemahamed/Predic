import * as vscode from 'vscode';
import { PredicAPIClient } from '../api/client';

export class PredicInlineCompletionProvider implements vscode.InlineCompletionItemProvider {
    private client: PredicAPIClient;

    constructor() {
        this.client = new PredicAPIClient();
    }

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[]> {
        
        // Debounce manually if needed, or rely on VS Code's internal triggering
        
        // Get text before cursor
        const textBefore = document.getText(new vscode.Range(new vscode.Position(0, 0), position));
        
        // Optional: Get text after cursor for FIM (Fill In Middle) if model supports it
        // For simple start, we just send prefix
        
        const prediction = await this.client.completion(textBefore);
        
        if (!prediction || prediction.trim() === "") {
            return [];
        }

        return [new vscode.InlineCompletionItem(prediction, new vscode.Range(position, position))];
    }
}