import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';

const CFG_NS = 'predic';
function cfg<T>(k: string, d: T): T {
  return vscode.workspace.getConfiguration(CFG_NS).get<T>(k, d);
}

class Bridge {
  constructor(private root: string) {}
  run(prompt: string): Promise<string> {
    return new Promise((res, rej) => {
      const py = cfg<string>('pythonPath', 'python3');
      const script = path.join(this.root, 'python', 'predic.py');
      const timeout = cfg<number>('inferenceTimeoutMs', 2500);
      cp.execFile(py, [script, prompt], { timeout }, (e, out, err) => {
        if (e) {return rej(err || e.message);}
        {res(out.trim());}
      });
    });
  }
}

class Provider implements vscode.InlineCompletionItemProvider {
  constructor(private bridge: Bridge, private sb: vscode.StatusBarItem) {}
  async provideInlineCompletionItems(
    doc: vscode.TextDocument,
    pos: vscode.Position,
    _c: vscode.InlineCompletionContext,
    token: vscode.CancellationToken
  ) {
    if (!cfg<boolean>('enabled', true) || token.isCancellationRequested) {return;}
    const full = doc.getText(new vscode.Range(0, 0, pos.line, pos.character));
    if (!full.trim()) {return;}
    try {
      this.sb.text = 'Predic $(sync~spin)';
      const out = await this.bridge.run(full);
      this.sb.text = 'Predic';
      if (!out) {return;}
      return [new vscode.InlineCompletionItem(out, new vscode.Range(pos, pos))];
    } catch {
      this.sb.text = 'Predic ⚠️';
    }
  }
}

export function activate(ctx: vscode.ExtensionContext) {
  const sb = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  sb.text = 'Predic';
  sb.show();
  ctx.subscriptions.push(sb);
  ctx.subscriptions.push(
    vscode.commands.registerCommand('predic.toggle', async () => {
      const cur = cfg<boolean>('enabled', true);
      await vscode.workspace.getConfiguration(CFG_NS).update('enabled', !cur, vscode.ConfigurationTarget.Global);
      vscode.window.showInformationMessage(`Predic ${!cur ? 'enabled' : 'disabled'}.`);
    })
  );
  const bridge = new Bridge(ctx.extensionPath);
  const langs = [{ language: 'javascriptreact' }, { language: 'typescriptreact' }, { language: 'css' }];
  ctx.subscriptions.push(vscode.languages.registerInlineCompletionItemProvider(langs, new Provider(bridge, sb)));

  console.log('Predic extension is now active!');
  // Status bar, commands etc.
}

export function deactivate() {
  console.log('Predic extension is now deactivated!');
}

