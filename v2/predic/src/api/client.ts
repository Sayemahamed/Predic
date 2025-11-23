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

    async chat(messages: any[]) {
        // KoboldCpp OpenAI-compatible endpoint
        return await axios.post(`${this.baseUrl}/v1/chat/completions`, {
            messages: messages,
            model: "koboldcpp", // Name doesn't matter for local
            max_tokens: 2048,
            stream: false 
        });
    }

    async completion(prompt: string) {
        const response = await axios.post(`${this.baseUrl}/v1/completions`, {
            prompt: prompt,
            max_tokens: 50,
            stop: ["\n"],
            temperature: 0.2
        });
        return response.data.choices[0].text;
    }
}