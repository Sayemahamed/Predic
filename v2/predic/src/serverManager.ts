import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { PredicAPIClient } from './api/client';

export class ServerManager {
    private serverProcess: ChildProcess | null = null;
    private outputChannel: vscode.OutputChannel;
    private apiClient: PredicAPIClient;
    private isStarting: boolean = false;

    constructor(private context: vscode.ExtensionContext) {
        this.outputChannel = vscode.window.createOutputChannel('Predic Server');
        this.apiClient = new PredicAPIClient();
    }

    async start(): Promise<boolean> {
        if (this.isStarting || await this.apiClient.checkHealth()) {
            return true;
        }

        this.isStarting = true;
        this.outputChannel.show();
        this.outputChannel.appendLine('Starting KoboldCpp server...');

        const config = vscode.workspace.getConfiguration('predic');
        
        // Resolve Paths (Auto-detect if not set)
        let exePath = config.get<string>('koboldCppPath');
        let modelPath = config.get<string>('modelPath');
        const port = config.get<number>('port') || 5001;
        const gpuLayers = config.get<number>('gpuLayers') || -1;
        const contextSize = config.get<number>('contextSize') || 8192;

        // Auto-resolve relative paths for development convenience
        if (!exePath || !modelPath) {
            const workspaceRoot = vscode.workspace.workspaceFolders?.[0].uri.fsPath || '';
            // Assuming v2 structure: workspace/koboldcpp.exe
            if (!exePath) exePath = path.join(workspaceRoot, '..', 'koboldcpp.exe'); 
            if (!modelPath) modelPath = path.join(workspaceRoot, '..', 'models', 'qwen2.5-coder-0.5B-Instruct-q4_k_m.gguf');
        }

        if (!fs.existsSync(exePath)) {
            vscode.window.showErrorMessage(`KoboldCpp executable not found at: ${exePath}`);
            this.isStarting = false;
            return false;
        }

        if (!fs.existsSync(modelPath)) {
            vscode.window.showErrorMessage(`Model file not found at: ${modelPath}`);
            this.isStarting = false;
            return false;
        }

        // Spawn the Process
        const args = [
            '--model', modelPath,
            '--port', port.toString(),
            '--gpulayers', gpuLayers.toString(),
            '--contextsize', contextSize.toString(),
            '--usecublas', // Enable CUDA if available
            '--smartcontext' 
        ];

        this.outputChannel.appendLine(`Executing: ${exePath} ${args.join(' ')}`);

        try {
            this.serverProcess = spawn(exePath, args);

            this.serverProcess.stdout?.on('data', (data) => {
                const msg = data.toString();
                this.outputChannel.append(msg);
                // KoboldCpp prints this when ready
                if (msg.includes('HTTP server is listening')) { 
                    this.isStarting = false;
                    vscode.window.showInformationMessage('Predic (KoboldCpp) Started!');
                }
            });

            this.serverProcess.stderr?.on('data', (data) => this.outputChannel.append(`ERR: ${data.toString()}`));
            
            this.serverProcess.on('close', (code) => {
                this.outputChannel.appendLine(`Server stopped with code ${code}`);
                this.serverProcess = null;
                this.isStarting = false;
            });

            // Wait a bit for startup
            return await this.waitForServer();

        } catch (err) {
            vscode.window.showErrorMessage(`Failed to spawn KoboldCpp: ${err}`);
            this.isStarting = false;
            return false;
        }
    }

    async stop() {
        if (this.serverProcess) {
            this.serverProcess.kill();
            this.serverProcess = null;
        }
    }

    private async waitForServer(): Promise<boolean> {
        for (let i = 0; i < 20; i++) { // Wait up to 20 seconds
            if (await this.apiClient.checkHealth()) return true;
            await new Promise(r => setTimeout(r, 1000));
        }
        return false;
    }
}