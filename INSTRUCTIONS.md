# 🛠️ Developer Guide for Predic

Welcome to the Predic codebase! This guide will help you set up your development environment to build, debug, and modify the extension.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
1.  **Node.js** (v18 or newer) & **npm**.
2.  **Visual Studio Code** (latest version).
3.  **KoboldCpp Executable**:
    -   This extension manages a backend process. You need the actual binary to test it.
    -   Download the latest release for your OS from [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp/releases).
    -   Rename it to `koboldcpp.exe` (Windows) or `koboldcpp` (Mac/Linux).

## 🏗️ Setup & Installation

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/Sayemahamed/Predic.git](https://github.com/Sayemahamed/Predic.git)
    cd Predic
    ```

2.  **Install Dependencies**:
    Run this command in the root folder to install VS Code types, Webpack, and runtime libraries (axios, openai):
    ```bash
    npm install
    ```

3.  **Backend Setup (Crucial Step)**:
    -   Create a `models/` folder in the project root.
    -   Download a small GGUF model (e.g., [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF)) and place it in `models/`.
    -   Place your downloaded `koboldcpp` executable in the project root (or configure the path in VS Code settings after launching).

## ▶️ Running in Debug Mode

1.  Open the project folder in VS Code.
2.  Press **F5** (or select `Run` > `Start Debugging` from the menu).
3.  A new window (**Extension Development Host**) will appear.
4.  **Verify**:
    -   Check the **Output Panel** (`Ctrl+Shift+U`) -> Select **"Predic Server"** from the dropdown. You should see "Starting KoboldCpp...".
    -   Click the **Predic icon** in the sidebar to open the Chat.

## 📦 Building the Extension (.vsix)

To create an installable file for distribution:

1.  Install the VS Code Extension Manager CLI:
    ```bash
    npm install -g @vscode/vsce
    ```
2.  Run the package command:
    ```bash
    vsce package
    ```
3.  This will generate `predic-0.0.3.vsix`. You can share this file or install it manually via the Extensions view -> `...` -> `Install from VSIX`.

## 🤝 Contribution

If you want to add features or fix bugs, please follow the steps given below:
 - Fork the repository.
 - Create your feature branch (git checkout -b feature/AmazingFeature).
 - Commit your changes (git commit -m 'Add some AmazingFeature').
 - Push to the branch (git push origin feature/AmazingFeature).
 - Open a Pull Request.

## 📂 Project Architecture

-   **`src/extension.ts`**: The entry point. Registers commands and activates the extension.
-   **`src/serverManager.ts`**: Handles spawning and killing the `koboldcpp` background process.
-   **`src/providers/`**:
    -   `chatViewProvider.ts`: Logic for the Chat Sidebar (Webview). Handles messages, context, and streaming.
    -   `modelManagerProvider.ts`: Logic for the Dashboard Webview. Handles file scanning and downloads.
    -   `inlineCompletionProvider.ts`: The "Ghost Text" provider. Connects to the local API to fetch code completions.
-   **`media/`**: Contains the HTML/CSS/JS assets for the Webviews (Chat & Dashboard).

## 🐛 Troubleshooting

* **"KoboldCpp not found"**: Check your `settings.json` or ensure the binary is in the root folder.
* **"Webview is blank"**: Run `npm run compile` to ensure the latest changes are built. Check the "Developer Tools" inside the Extension Host (`Help` > `Toggle Developer Tools`).
* **Streaming Issues**: Ensure `openai` or `axios` is installed in `node_modules`.
