import { build } from 'esbuild';
import fs from 'fs-extra';
import path from 'path';

const isWatchMode = process.argv.includes('--watch');
const outDir = 'dist';
const modelSrc = path.join('src', 'model');
const modelDest = path.join(outDir, 'model');

const buildOptions = {
    entryPoints: ['src/extension.ts'],
    bundle: true,
    outfile: path.join(outDir, 'extension.js'),
    platform: 'node',
    target: 'node16',
    // This is the critical fix. It tells the bundler not to package these
    // native modules, which resolves the ".node" file error.
    external: ['vscode', 'onnxruntime-node', 'sharp'],
    sourcemap: true,
};

async function runBuild() {
    try {
        if (isWatchMode) {
            buildOptions.watch = {
                onRebuild(error) {
                    if (error) console.error('Watch build failed:', error);
                    else console.log('Rebuild succeeded.');
                },
            };
        }
        
        await build(buildOptions);
        console.log('Build finished successfully.');

        // Copy the model files into the final package.
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
