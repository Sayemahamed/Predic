//@ts-check
'use strict';

const path = require('path');
const webpack = require('webpack');

/**
 * @param {any} env
 * @param {{ mode: 'production' | 'development' }} argv
 * @returns {import('webpack').Configuration[]}
 */
module.exports = (env, argv) => {
  const isDevelopment = argv.mode === 'development';

  // Common configuration
  const commonConfig = {
    target: 'node',
    devtool: isDevelopment ? 'source-map' : false,
    resolve: {
      extensions: ['.ts', '.js'],
      mainFields: ['main', 'module'],
    },
    module: {
      rules: [
        {
          test: /\.ts$/,
          exclude: /node_modules/,
          use: [
            {
              loader: 'ts-loader',
              options: {
                transpileOnly: isDevelopment,
                compilerOptions: {
                  sourceMap: isDevelopment,
                },
              },
            },
          ],
        },
      ],
    },
    stats: isDevelopment ? 'minimal' : 'normal',
  };

  // Extension configuration
  const extensionConfig = {
    ...commonConfig,
    name: 'extension',
    entry: './src/extension.ts',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'extension.js',
      libraryTarget: 'commonjs2',
      devtoolModuleFilenameTemplate: '../[resource-path]',
    },
    externals: {
      vscode: 'commonjs vscode',
    },
    plugins: [
      new webpack.DefinePlugin({
        'process.env.NODE_ENV': JSON.stringify(argv.mode || 'development'),
      }),
    ],
  };

  // Agent configuration
  const agentConfig = {
    ...commonConfig,
    name: 'agent',
    entry: './src/agent.ts',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'agent.js',
      libraryTarget: 'commonjs2',
      devtoolModuleFilenameTemplate: '../[resource-path]',
    },
    externals: {
      // Externalize the entire transformers package and its dependencies
      '@xenova/transformers': 'commonjs @xenova/transformers',
      'onnxruntime-node': 'commonjs onnxruntime-node',
      'sharp': 'commonjs sharp',
      // Node.js built-ins
      child_process: 'commonjs child_process',
      fs: 'commonjs fs',
      path: 'commonjs path',
      os: 'commonjs os',
      crypto: 'commonjs crypto',
      stream: 'commonjs stream',
      util: 'commonjs util',
      events: 'commonjs events',
      http: 'commonjs http',
      https: 'commonjs https',
      url: 'commonjs url',
      zlib: 'commonjs zlib',
    },
    plugins: [
      new webpack.DefinePlugin({
        'process.env.NODE_ENV': JSON.stringify(argv.mode || 'development'),
      }),
      // Ignore native modules and binaries
      new webpack.IgnorePlugin({
        resourceRegExp: /\.node$/,
      }),
      new webpack.IgnorePlugin({
        resourceRegExp: /^fsevents$/,
      }),
    ],
    optimization: {
      minimize: !isDevelopment,
    },
    node: {
      __dirname: false,
      __filename: false,
    },
  };

  return [extensionConfig, agentConfig];
};