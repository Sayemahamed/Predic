import { AutoTokenizer } from "@huggingface/transformers";

// Load tokenizer for a gated repository.
const tokenizer = await AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B");

// Encode text.
const text = "Hello world!";
const encoded = tokenizer.encode(text);
console.log(encoded);
const decoded = tokenizer.decode(encoded);
console.log(decoded);
