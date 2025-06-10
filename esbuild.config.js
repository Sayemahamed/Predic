const { build } = require('esbuild');

build({
  entryPoints: ['src/extension.ts'],
  bundle: true,
  outfile: 'dist/extension.js',
  platform: 'node',
  target: 'node16',
  // The 'vscode' module is provided by the runtime
  // Mark onnxruntime-node as external to prevent bundling of native .node files
  external: ['vscode', 'onnxruntime-node' , 'sharp'], 
  sourcemap: true,
}).catch(() => process.exit(1));
