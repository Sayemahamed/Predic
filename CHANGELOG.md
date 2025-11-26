# Change Log

All notable changes to the "Predic" extension will be documented in this file.

## [0.0.2] - 2023-12-27
### Major Architecture Overhaul
-   **Engine Swap**: Replaced the limited `transformers.js` in-process backend with a robust **KoboldCpp** server integration. This enables support for larger, state-of-the-art GGUF models (Qwen, DeepSeek, Llama 3) and GPU acceleration.
-   **New Chat UI**:
    -   Complete redesign matching native VS Code aesthetics.
    -   Added **Streaming Responses** for real-time feedback.
    -   Added **@filename context** support (auto-finds and attaches files).
    -   Added User Message actions (Edit/Copy).
    -   Added input box syntax highlighting.
-   **Dashboard**: Introduced a graphical "Model Manager" to download, list, and switch models easily.
-   **Inline Completion**: Updated "Ghost Text" provider to use **FIM (Fill-In-the-Middle)** for smarter, context-aware suggestions.
-   **Performance**: Moved heavy inference out of the extension host process to a dedicated local server.

## [0.0.1] - 2023-10-01
### Initial Release
-   **Proof of Concept**: Basic implementation using [Xenova/transformers.js](https://github.com/xenova/transformers.js) running ONNX models directly in the extension host.
-   **Inline Completion**: Simple, single-line code suggestions for JavaScript files.
-   **Official Models**: Added support for downloading ReaComplete, our official fine-tuned model.
-   **Offline First**: Demonstrated core capability of running inference without internet access.
