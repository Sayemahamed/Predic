const path = require('path');
console.log(path);
const fs = require('fs');

let isInitialized = false;

/**
 * Initializes the environment for the transformers library.
 * This MUST be called before any other function in this file.
 * @param {string} cachePath - The absolute path to the extension's storage folder.
 */
async function init(cachePath) {
    if (isInitialized) return;
    if (!cachePath || typeof cachePath !== 'string') {
        throw new Error('A valid storage path was not provided for initialization.');
    }

    const modelsPath = path.join(cachePath, 'models');
    process.env.TRANSFORMERS_CACHE = modelsPath;

    try {
        if (!fs.existsSync(modelsPath)) {
            await fs.promises.mkdir(modelsPath, { recursive: true });
        }
        console.log(`Transformers.js cache path set and verified: ${modelsPath}`);
        isInitialized = true;
    } catch (error) {
        console.error(`Failed to create cache directory at ${modelsPath}:`, error);
        throw new Error(`Could not create storage directory for models. Please check folder permissions.`);
    }
}

let pipelineInstance = null;

class GemmaPipeline {
    static task = 'text-generation';
    static model = 'onnx-community/gemma-3-1b-it-ONNX-GQA';

    // Simplified getInstance to be a direct async function for clearer error tracking.
    static async getInstance(progress_callback) {
        if (pipelineInstance === null) {
            try {
                console.log('Pipeline instance is null, creating new one...');
                const { pipeline } = await import('@xenova/transformers');
                pipelineInstance = await pipeline(this.task, this.model, {
                    quantization: 'q4',
                    progress_callback,
                });
                console.log('Pipeline instance created successfully.');
            } catch (error) {
                console.error('Error creating pipeline instance:', error);
                // Throw a more specific error to be caught by the activate function.
                throw new Error(`Failed to download or load the AI model. Details: ${error.message}`);
            }
        }
        return pipelineInstance;
    }
}

const runGemma = async (prompt) => {
    // Pass a logging function as the progress_callback to see download status
    const generator = await GemmaPipeline.getInstance(progress => {
        console.log(`Model loading progress: ${JSON.stringify(progress)}`);
    });
    
    const messages = [
        { role: "system", content: "You are a helpful code completion assistant. Complete the user's code. Provide only the code completion, without any explanation or repeating the user's code." },
        { role: "user", content: prompt },
    ];
    
    const output = await generator(messages, {
        max_new_tokens: 64,
        do_sample: true,
        temperature: 0.7,
        top_p: 0.9,
    });

    const lastMessage = output[0].generated_text.at(-1);
    return lastMessage ? lastMessage.content.trim() : '';
};

module.exports = {
    init,
    runGemma,
};
