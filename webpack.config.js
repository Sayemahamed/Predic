//@ts-check
'use strict';
const path = require('path');

/**@type {import('webpack').Configuration}*/
const config = {
  target: 'node',
  entry: {
    extension: './src/extension.ts', // Your main extension file
    agent: './src/agent.ts'          // Your new agent file
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].js', // This creates 'extension.js' and 'agent.js'
    libraryTarget: 'commonjs2',
    devtoolModuleFilenameTemplate: '../[resource-path]',
  },
  devtool: 'source-map',
  externals: {
    vscode: 'commonjs vscode',
    child_process: 'commonjs child_process'
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        exclude: /node_modules/,
        use: [ { loader: 'ts-loader' } ],
      },
    ],
  },
};
module.exports = config;