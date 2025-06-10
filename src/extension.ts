import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {

    // console.log('Congratulations, your extension "Predic" is active with multi-context capabilities!');

    const provider: vscode.InlineCompletionItemProvider = {
        
        async provideInlineCompletionItems(document: vscode.TextDocument, position: vscode.Position, context: vscode.InlineCompletionContext, token: vscode.CancellationToken): Promise<vscode.InlineCompletionItem[] | undefined> {
            
            // Input #1: The content on the current line before the cursor
            const line = document.lineAt(position.line);
            const textBeforeCursorOnLine = line.text.substring(0, position.character);

            // Input #2: The entire file content from the start up to the cursor
            const rangeBeforeCursor = new vscode.Range(new vscode.Position(0, 0), position);
            const textBeforeCursorInFile = document.getText(rangeBeforeCursor);

            // Input #3: The currently selected text by the user
            const editor = vscode.window.activeTextEditor;
            let selectedText = '';
            if (editor && !editor.selection.isEmpty) {
                // Get text from the active editor's selection
                selectedText = document.getText(editor.selection);
            }

            // For debugging: 
            // console.log("--- Predic Context ---");
            // console.log("1. Line Before Cursor:", textBeforeCursorOnLine);
            // console.log("2. File Before Cursor:", textBeforeCursorInFile);
            // console.log("3. Selected Text:", selectedText);
            // console.log("----------------------");


            // --- 2. GET SUGGESTION FROM YOUR MODEL (Placeholder Logic) ---
            
            // The logic here is just for demonstration.
            let suggestion = '';

            // Example: If user has selected text, suggest wrapping it in a div
            if (selectedText) {
                suggestion = `<div>\n\t${selectedText}\n</div>`;
                
                return [new vscode.InlineCompletionItem(suggestion)];
            }
            
            // Fallback to the previous logic if nothing is selected
            if (textBeforeCursorOnLine.trim().endsWith('const name =')) {
                suggestion = ' "Predic";';
            } else if (textBeforeCursorOnLine.trim().endsWith('<div>')) {
                suggestion = '<h1>Hello from Predic!</h1></div>';
            } else if (textBeforeCursorOnLine.trim().endsWith('className="')) {
                suggestion = 'flex items-center justify-center">';
            } else {
                return; // No suggestion
            }
            
            return [new vscode.InlineCompletionItem(suggestion)];
        },
    };

    // Register the provider for all languages
    vscode.languages.registerInlineCompletionItemProvider({ pattern: '**' }, provider);
}

export function deactivate() {}
