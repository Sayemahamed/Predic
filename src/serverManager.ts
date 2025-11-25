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
        
        let exePath = config.get<string>('koboldCppPath');
        let modelPath = config.get<string>('modelPath');
        const port = config.get<number>('port') || 5001;
        const gpuLayers = config.get<number>('gpuLayers'); // Don't force default here, handle below
        const contextSize = config.get<number>('contextSize') || 8192;

        // Path resolution logic (same as before)
        if (!exePath || !modelPath) {
            const extensionRoot = this.context.extensionUri.fsPath;
            if (!exePath) exePath = path.join(extensionRoot, '..', 'koboldcpp.exe');
            if (!modelPath) modelPath = path.join(extensionRoot, '..', 'models', 'qwen2.5-coder-0.5B-Instruct-q4_k_m.gguf');
        }

        if (!fs.existsSync(exePath)) {
            vscode.window.showErrorMessage(`KoboldCpp executable not found at: ${exePath}`);
            this.isStarting = false;
            return false;
        }

        // Construct Arguments matching your working manual command
        const args = [
            '--model', modelPath,
            '--port', port.toString(),
            '--contextsize', contextSize.toString(),
            '--smartcontext' 
        ];

        // Only add GPU layers if explicitly set to a positive number
        // We avoid sending '-1' (auto) because it crashed your specific setup
        if (gpuLayers && gpuLayers > 0) {
            args.push('--gpulayers', gpuLayers.toString());
        }

        this.outputChannel.appendLine(`Executing: "${exePath}" ${args.join(' ')}`);

        try {
            this.serverProcess = spawn(exePath, args, {
                cwd: path.dirname(exePath)
            });

            this.serverProcess.stdout?.on('data', (data) => {
                const msg = data.toString();
                this.outputChannel.append(msg);
                if (msg.includes('HTTP server is listening') || msg.includes('network mode')) { 
                    this.isStarting = false;
                    vscode.window.showInformationMessage('Predic (KoboldCpp) Started!');
                }
            });

            this.serverProcess.stderr?.on('data', (data) => {
                this.outputChannel.append(`LOG: ${data.toString()}`);
            });
            
            this.serverProcess.on('close', (code) => {
                this.outputChannel.appendLine(`Server stopped with code ${code}`);
                this.serverProcess = null;
                this.isStarting = false;
            });

            return await this.waitForServer();

        } catch (err: any) {
            vscode.window.showErrorMessage(`Failed to spawn KoboldCpp: ${err.message}`);
            this.isStarting = false;
            return false;
        }
    }
    
    async stop() {
        if (this.serverProcess) {
            this.serverProcess.kill();
            this.serverProcess = null;
            this.outputChannel.appendLine('Server killed manually.');
        }
    }

    private async waitForServer(): Promise<boolean> {
        for (let i = 0; i < 20; i++) { // Wait up to 20 seconds
            if (await this.apiClient.checkHealth()) return true;
            await new Promise(r => setTimeout(r, 1000));
        }
        this.outputChannel.appendLine('Timed out waiting for server health check.');
        return false;
    }
}