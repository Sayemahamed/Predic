export interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

export interface Model {
    id: string;
    name: string;
    size: string;
    status: 'ready' | 'downloading' | 'missing';
    path?: string;
    url?: string; // URL for downloading
}