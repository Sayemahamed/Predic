import axios from 'axios';
import * as vscode from 'vscode';

export class PredicAPIClient {
    private get baseUrl(): string {
        const config = vscode.workspace.getConfiguration('predic');
        const port = config.get<number>('port') || 5001;
        return `http://localhost:${port}`;
    }

    async checkHealth(): Promise<boolean> {
        try {
            await axios.get(`${this.baseUrl}/api/extra/version`, { timeout: 2000 });
            return true;
        } catch (error) {
            return false;
        }
    }

    async chatStream(messages: any[], onToken: (token: string) => void): Promise<void> {
        try {
            const response = await axios.post(`${this.baseUrl}/v1/chat/completions`, {
                messages: messages,
                model: "koboldcpp",
                max_tokens: 2048,
                stream: true
            }, {
                responseType: 'stream' // Crucial: Get the raw stream
            });

            const stream = response.data;
            let buffer = '';

            // Process chunks as they arrive
            for await (const chunk of stream) {
                // 1. Append new chunk to buffer
                buffer += chunk.toString();

                // 2. Process all complete lines in the buffer
                const lines = buffer.split('\n');
                
                // The last line might be incomplete, so we save it back to the buffer
                // to wait for the next chunk.
                buffer = lines.pop() || ''; 

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || trimmed === 'data: [DONE]') continue;

                    if (trimmed.startsWith('data: ')) {
                        try {
                            const jsonStr = trimmed.substring(6); // Remove "data: "
                            const json = JSON.parse(jsonStr);
                            const content = json.choices[0]?.delta?.content;
                            
                            if (content) {
                                onToken(content);
                            }
                        } catch (e) {
                            // If a line is malformed, we log it but don't crash
                            console.warn("Skipping malformed line:", trimmed);
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Stream Error:", error);
            throw error; // Re-throw so the UI knows to show an error
        }
    }
}