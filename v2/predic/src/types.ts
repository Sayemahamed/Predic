export interface Model {
    id: string;
    name: string;
    size: string;
    status: 'available' | 'downloading' | 'ready' | 'error';
    progress: number;
    description?: string;
    size_category?: 'small' | 'medium' | 'large';
}

export interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp?: number;
}

export interface CompletionRequest {
    model_id: string;
    prompt: string;
    max_tokens?: number;
    temperature?: number;
    stop_sequences?: string[];
}

export interface ChatRequest {
    model_id: string;
    messages: ChatMessage[];
    max_tokens?: number;
    temperature?: number;
}