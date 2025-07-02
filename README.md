# 🔮 Predic: Offline & Private AI Code Completion

<p align="center">
  <img src="https://raw.githubusercontent.com/Sayemahamed/Predic/main/media/logo.png" width="120" />
</p>

<p align="center">
  <strong>Your Private and Offline AI code Completion Agent</strong>
</p>

[![VS Code Marketplace](https://img.shields.io/visual-studio-marketplace/v/your-publisher.predic?style=for-the-badge&label=Marketplace&color=blue)](https://marketplace.visualstudio.com/items?itemName=your-publisher.predic)
[![Installs](https://img.shields.io/visual-studio-marketplace/i/your-publisher.predic?style=for-the-badge&color=green)](https://marketplace.visualstudio.com/items?itemName=your-publisher.predic)
[![Rating](https://img.shields.io/visual-studio-marketplace/r/your-publisher.predic?style=for-the-badge&color=yellow)](https://marketplace.visualstudio.com/items?itemName=your-publisher.predic)

**Predic** is your personal AI pair programmer that runs entirely on your local machine. Get helpful code completions without ever sending your code to the cloud. It's fast, private, and works completely offline.

---

![Predic Demo GIF](https://raw.githubusercontent.com/Sayemahamed/Predic/main/media/predic-demo.mp4)
<!-- > *(**Pro-tip:** Create a cool animated GIF showing Predic in action and replace the link above!)* -->

## ✨ Core Features

* **100% Offline:** After a one-time model download, Predic works without an internet connection.
* **Completely Private:** Your code never leaves your computer. Period. We value your privacy and intellectual property.
* **Instant Completions:** Uses a lightweight, quantized AI model that runs efficiently on your local hardware.
* **Multi-Language Support:** Get smart completions for modern web development.
    * JavaScript & TypeScript
    * React (.js, .ts, .jsx & .tsx)
    * CSS
* **Easy to Use:** Simply install from the marketplace and start coding. No complex setup or API keys required.

## 🚀 Getting Started

1.  Install the extension from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=your-publisher.predic).
2.  The first time it runs, Predic will download the AI model (approx. 150-200MB). You can see the progress in the status bar.
3.  Once the status bar shows **`$(zap) Predic: Ready`**, you're all set! Open a supported file and start coding to see suggestions.

## 🛠️ How It Works

Predic uses a modern, two-process architecture to ensure the VS Code UI remains fast and responsive.

1.  **Extension Host:** The main extension (`extension.ts`) interacts with the VS Code API, captures your code context, and displays suggestions.
2.  **Agent Process:** A separate Node.js process (`agent.ts`) is forked to handle the heavy lifting. This agent loads a quantized AI model using the amazing [Transformers.js](https://github.com/xenova/transformers.js) library and runs inference without blocking the main editor thread.

This design ensures that even during intense AI computations, your typing and editing experience remains silky smooth.

## 🔧 Contributing & Local Development

We welcome contributions! Whether it's a bug report, a feature request, or a pull request, we'd love your help.

To get started with local development:

1.  **Prerequisites:** Make sure you have [Node.js](https://nodejs.org/) installed.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Sayemahamed/Predic.git
    cd Predic
    ```
3.  **Install Dependencies:**
    ```bash
    npm install
    ```
4.  **Start the Development Build:**
    ```bash
    npm run watch
    ```
5.  **Run the Extension:**
    * Press `F5` in VS Code to open a new Extension Development Host window.
    * This window will have the Predic extension loaded and ready to debug.
    * Logs and debugging output will appear in the **Debug Console** of your main VS Code window.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgements

* A huge thank you to the team behind [Xenova/Transformers.js](https://github.com/xenova/transformers.js) for making local, in-browser, and in-app AI accessible to everyone.

