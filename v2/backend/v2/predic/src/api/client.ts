import axios, { AxiosInstance } from 'axios';
import * as vscode from 'vscode';
import { Model, CompletionRequest, ChatRequest, ChatMessage } from '../types';

export class PredicAPIClient {
    private client: AxiosInstance;
    private serverUrl: string;

    constructor() {
        const config = vscode.workspace.getConfiguration('predic');
        this.serverUrl = config.get('serverUrl', 'http://localhost:8000');
        
        this.client = axios.create({
            baseURL: this.serverUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
            },
        });
    }

    async checkHealth(): Promise<boolean> {
        try {
            const response = await this.client.get('/health');
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }

    async getAvailableModels(): Promise<Model[]> {
        try {
            const response = await this.client.get('/api/models/available');
            return response.data.models;
        } catch (error) {
            console.error('Failed to fetch models:', error);
            throw error;
        }
    }

    async downloadModel(modelId: string): Promise<void> {
        try {
            await this.client.post(`/api/models/download/${modelId}`);
        } catch (error) {
            console.error('Failed to download model:', error);
            throw error;
        }
    }

    async getModelStatus(modelId: string): Promise<any> {
        try {
            const response = await this.client.get(`/api/models/status/${modelId}`);
            return response.data;
        } catch (error) {
            console.error('Failed to get model status:', error);
            throw error;
        }
    }

    async generateCompletion(request: CompletionRequest): Promise<string> {
        try {
            const response = await this.client.post('/api/completion/', request);
            return response.data.completion;
        } catch (error) {
            console.error('Failed to generate completion:', error);
            throw error;
        }
    }

    async chat(request: ChatRequest): Promise<ChatMessage> {
        try {
            const response = await this.client.post('/api/chat/', request);
            return response.data.message;
        } catch (error) {
            console.error('Failed to send chat message:', error);
            throw error;
        }
    }
}