from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# --- Basic Server and Model Setup ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

MODEL_NAME = "SmolLM/SmolLM-350M-Instruct"
PORT = 5112

# --- Load the Model and Tokenizer ---
try:
    print(f"--- Loading tokenizer for {MODEL_NAME}... ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"--- Loading model {MODEL_NAME}... ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    print("--- Model loaded successfully. Server is ready. ---")
except Exception as e:
    print(f"FATAL: Could not load model. Error: {e}")
    exit()

# --- API Endpoint for Code Completion ---
@app.route('/complete', methods=['POST'])
def complete():
    try:
        data = request.get_json()
        prompt_text = data.get('prompt')
        if not prompt_text:
            return jsonify({'error': 'Prompt is required'}), 400

        messages = [{"role": "user", "content": f"Complete the following code snippet:\n```\n{prompt_text}\n```"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=64)
        
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        suggestion = completion.split("```")[-1].strip()

        return jsonify({'suggestion': suggestion})
    except Exception as e:
        print(f"Error during completion request: {e}")
        return jsonify({'error': 'Failed to generate completion'}), 500

# --- Start the Server ---
if __name__ == '__main__':
    print(f"--- SmolLM server starting on [http://127.0.0.1](http://127.0.0.1):{PORT} ---")
    app.run(host='0.0.0.0', port=PORT, debug=False)