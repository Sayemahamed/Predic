const { build } = require('esbuild');
const fs = require('fs-extra');
const path = require('path');

const isWatchMode = process.argv.includes('--watch');
const outDir = 'dist';

// Define the common build options for esbuild.
const buildOptions = {
    entryPoints: ['src/extension.ts'],
    bundle: true,
    outfile: path.join(outDir, 'extension.js'),
    platform: 'node',
    target: 'node16',
    external: ['vscode', 'onnxruntime-node', 'sharp'],
    sourcemap: true,
};

async function runBuild() {
    try {
        // Conditionally add the 'watch' configuration if in watch mode.
        // This prevents the "Invalid option" error when running a normal build.
        if (isWatchMode) {
            buildOptions.watch = {
                onRebuild(error) {
                    if (error) console.error('Watch build failed:', error);
                    else console.log('Rebuild succeeded.');
                },
            };
            console.log('Running in watch mode...');
        }

        // 1. Bundle the extension code using the configured options.
        await build(buildOptions);
        console.log('Build finished successfully.');

        // 2. Copy the model files to the output directory.
        const modelSrc = path.join('src', 'model');
        const modelDest = path.join(outDir, 'model');
        if (fs.existsSync(modelSrc)) {
            fs.copySync(modelSrc, modelDest, { overwrite: true });
            console.log('Model files copied successfully.');
        }

    } catch (e) {
        console.error('Build process failed:', e);
        process.exit(1);
    }
}

runBuild();
