import * as vscode from 'vscode';
import { spawn, ChildProcess, exec } from 'child_process';
import * as path from 'path';
import { PredicAPIClient } from './api/client';

export class ServerManager {
    private serverProcess: ChildProcess | null = null;
    private outputChannel: vscode.OutputChannel;
    private apiClient: PredicAPIClient;
    private isStarting: boolean = false;
    private lastHealthStatus: boolean = false;
    private _onStatusChanged: vscode.EventEmitter<boolean> = new vscode.EventEmitter<boolean>();
    public readonly onStatusChanged: vscode.Event<boolean> = this._onStatusChanged.event;

    constructor(private context: vscode.ExtensionContext) {
        this.outputChannel = vscode.window.createOutputChannel('Predic Server');
        this.apiClient = new PredicAPIClient();
        
        // Check server status periodically
        this.checkServerStatus();
        setInterval(() => this.checkServerStatus(), 30000);
    }

    async start(): Promise<boolean> {
        if (this.isStarting) {
            vscode.window.showInformationMessage('Server is already starting');
            return true;
        }

        this.isStarting = true;
        this.outputChannel.show();
        this.outputChannel.appendLine('Starting Predic backend server...');

        try {
            // Check if server is already running
            const isHealthy = await this.apiClient.checkHealth();
            if (isHealthy) {
                this.outputChannel.appendLine('Server is already running');
                vscode.window.showInformationMessage('Predic server is already running');
                this.isStarting = false;
                this.lastHealthStatus = true;
                this._onStatusChanged.fire(true);
                return true;
            }

            // Get backend path from configuration or try to find it
            const config = vscode.workspace.getConfiguration('predic');
            let backendPath = config.get<string>('backendPath', '');
            
            if (!backendPath) {
                // Try to auto-detect backend path
                const possiblePaths = [
                    // Check if backend is in workspace
                    ...(vscode.workspace.workspaceFolders || []).map(folder => 
                        path.join(folder.uri.fsPath, 'backend')
                    ),
                    // Check relative to extension
                    path.join(path.dirname(this.context.extensionPath), 'backend'),
                    // Check in parent directory
                    path.join(path.dirname(path.dirname(this.context.extensionPath)), 'backend'),
                    // Your specific path
                    'C:\\Projects\\Predic\\v2\\backend'
                ];
                
                this.outputChannel.appendLine('Auto-detecting backend path...');
                for (const testPath of possiblePaths) {
                    this.outputChannel.appendLine(`Checking: ${testPath}`);
                    if (require('fs').existsSync(testPath) && require('fs').existsSync(path.join(testPath, 'run.py'))) {
                        backendPath = testPath;
                        this.outputChannel.appendLine(`Found backend at: ${backendPath}`);
                        break;
                    }
                }
            }
            
            if (!backendPath || !require('fs').existsSync(backendPath)) {
                // Ask user to configure the path
                const action = await vscode.window.showErrorMessage(
                    'Backend directory not found. Please configure the backend path.',
                    'Configure Path',
                    'Browse...'
                );
                
                if (action === 'Browse...') {
                    const selected = await vscode.window.showOpenDialog({
                        canSelectFolders: true,
                        canSelectFiles: false,
                        canSelectMany: false,
                        openLabel: 'Select Backend Directory',
                        title: 'Select Predic Backend Directory'
                    });
                    
                    if (selected && selected[0]) {
                        backendPath = selected[0].fsPath;
                        await config.update('backendPath', backendPath, vscode.ConfigurationTarget.Global);
                    } else {
                        throw new Error('No backend directory selected');
                    }
                } else if (action === 'Configure Path') {
                    await vscode.commands.executeCommand('workbench.action.openSettings', 'predic.backendPath');
                    throw new Error('Please configure the backend path in settings');
                } else {
                    throw new Error('Backend directory not configured');
                }
            }
            
            this.outputChannel.appendLine(`Using backend path: ${backendPath}`);
            
            // Try different methods to start the server
            const started = await this.tryStartServer(backendPath);
            
            if (started) {
                vscode.window.showInformationMessage('Predic server started successfully');
                this.isStarting = false;
                this.lastHealthStatus = true;
                this._onStatusChanged.fire(true);
                return true;
            } else {
                throw new Error('Failed to start server with all methods');
            }

        } catch (error) {
            this.outputChannel.appendLine(`Failed to start server: ${error}`);
            vscode.window.showErrorMessage(`Failed to start Predic server: ${error}`);
            this.isStarting = false;
            this.lastHealthStatus = false;
            this._onStatusChanged.fire(false);
            return false;
        }
    }

    private async tryStartServer(backendPath: string): Promise<boolean> {
        // Method 1: Try with uv directly
        try {
            this.outputChannel.appendLine('Attempting to start with uv...');
            const success = await this.startWithUv(backendPath);
            if (success) return true;
        } catch (error) {
            this.outputChannel.appendLine(`UV method failed: ${error}`);
        }

        // Method 2: Try with python directly
        try {
            this.outputChannel.appendLine('Attempting to start with python directly...');
            const success = await this.startWithPython(backendPath);
            if (success) return true;
        } catch (error) {
            this.outputChannel.appendLine(`Python method failed: ${error}`);
        }

        // Method 3: Try activating venv and running
        try {
            this.outputChannel.appendLine('Attempting to start with venv activation...');
            const success = await this.startWithVenv(backendPath);
            if (success) return true;
        } catch (error) {
            this.outputChannel.appendLine(`Venv method failed: ${error}`);
        }

        return false;
    }

    private async startWithUv(backendPath: string): Promise<boolean> {
        const runScript = path.join(backendPath, 'run.py');
        
        // Check if uv exists
        const uvExists = await this.commandExists('uv');
        if (!uvExists) {
            throw new Error('uv not found in PATH');
        }

        this.serverProcess = spawn('uv', ['run', 'python', runScript], {
            cwd: backendPath,
            env: { ...process.env },
            shell: true
        });

        return await this.setupProcessAndWait();
    }

    private async startWithPython(backendPath: string): Promise<boolean> {
        const runScript = path.join(backendPath, 'run.py');
        
        // Try different python commands
        const pythonCommands = ['python', 'python3', 'py'];
        
        for (const cmd of pythonCommands) {
            try {
                const exists = await this.commandExists(cmd);
                if (exists) {
                    this.outputChannel.appendLine(`Trying with ${cmd}...`);
                    
                    this.serverProcess = spawn(cmd, [runScript], {
                        cwd: backendPath,
                        env: {
                            ...process.env,
                            PYTHONPATH: path.join(backendPath, 'src')
                        },
                        shell: true
                    });

                    const success = await this.setupProcessAndWait();
                    if (success) return true;
                }
            } catch (error) {
                continue;
            }
        }
        
        throw new Error('No python command found');
    }

    private async startWithVenv(backendPath: string): Promise<boolean> {
        const venvPath = path.join(backendPath, '.venv');
        const runScript = path.join(backendPath, 'run.py');
        
        if (!require('fs').existsSync(venvPath)) {
            throw new Error('No virtual environment found');
        }

        let activateScript: string;
        let command: string;
        
        if (process.platform === 'win32') {
            activateScript = path.join(venvPath, 'Scripts', 'activate.bat');
            command = `"${activateScript}" && python "${runScript}"`;
        } else {
            activateScript = path.join(venvPath, 'bin', 'activate');
            command = `source "${activateScript}" && python "${runScript}"`;
        }

        this.serverProcess = spawn(command, [], {
            cwd: backendPath,
            env: { ...process.env },
            shell: true
        });

        return await this.setupProcessAndWait();
    }

    private async setupProcessAndWait(): Promise<boolean> {
        if (!this.serverProcess) return false;

        this.serverProcess.stdout?.on('data', (data) => {
            this.outputChannel.append(data.toString());
        });

        this.serverProcess.stderr?.on('data', (data) => {
            this.outputChannel.append(data.toString());
        });

        this.serverProcess.on('error', (error) => {
            this.outputChannel.appendLine(`Process error: ${error}`);
        });

        this.serverProcess.on('close', (code) => {
            this.outputChannel.appendLine(`Server process exited with code ${code}`);
            this.serverProcess = null;
            this.lastHealthStatus = false;
            this._onStatusChanged.fire(false);
        });

        // Wait for server with shorter initial timeout
        try {
            await this.waitForServer(30);
            return true;
        } catch (error) {
            if (this.serverProcess) {
                this.serverProcess.kill();
                this.serverProcess = null;
            }
            return false;
        }
    }

    private async commandExists(command: string): Promise<boolean> {
        return new Promise((resolve) => {
            const checkCommand = process.platform === 'win32' ? 'where' : 'which';
            exec(`${checkCommand} ${command}`, (error) => {
                resolve(!error);
            });
        });
    }

    private async waitForServer(maxAttempts: number = 30): Promise<void> {
        this.outputChannel.appendLine(`Waiting for server to be ready...`);
        
        // Give the server a moment to start
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        for (let i = 0; i < maxAttempts; i++) {
            try {
                const isHealthy = await this.apiClient.checkHealth();
                if (isHealthy) {
                    this.outputChannel.appendLine('Server is ready!');
                    return;
                }
            } catch (error) {
                // Server not ready yet
            }
            
            // Check if process died
            if (this.serverProcess && this.serverProcess.exitCode !== null) {
                throw new Error(`Server process exited with code ${this.serverProcess.exitCode}`);
            }
            
            if (i % 5 === 0 && i > 0) {
                this.outputChannel.appendLine(`Still waiting... (${i}/${maxAttempts})`);
            }
            
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        throw new Error('Server failed to start within timeout');
    }

    async stop(): Promise<void> {
        if (!this.serverProcess) {
            const isHealthy = await this.apiClient.checkHealth();
            if (isHealthy) {
                vscode.window.showInformationMessage('Server is running externally and cannot be stopped from VS Code');
                return;
            }
            
            vscode.window.showInformationMessage('Server is not running');
            return;
        }

        this.outputChannel.appendLine('Stopping Predic backend server...');
        
        if (process.platform === 'win32') {
            spawn('taskkill', ['/pid', this.serverProcess.pid!.toString(), '/f', '/t'], { shell: true });
        } else {
            this.serverProcess.kill('SIGTERM');
        }
        
        this.serverProcess = null;
        this.lastHealthStatus = false;
        this._onStatusChanged.fire(false);
        
        vscode.window.showInformationMessage('Predic server stopped');
    }

    async checkServerStatus(): Promise<boolean> {
        try {
            const isHealthy = await this.apiClient.checkHealth();
            this.lastHealthStatus = isHealthy;
            this._onStatusChanged.fire(isHealthy);
            return isHealthy;
        } catch (error) {
            this.lastHealthStatus = false;
            this._onStatusChanged.fire(false);
            return false;
        }
    }

    isRunning(): boolean {
        return this.serverProcess !== null || this.lastHealthStatus;
    }

    async isServerHealthy(): Promise<boolean> {
        try {
            return await this.apiClient.checkHealth();
        } catch (error) {
            return false;
        }
    }
}