import { AutoTokenizer } from "@huggingface/transformers";

// Load tokenizer for a gated repository.
const tokenizer = await AutoTokenizer.from_pretrained(
  "meta-llama/Llama-2-7b-hf"
);

// Encode text.
const text = "Hello world!";
const encoded = tokenizer.encode(text);
console.log(encoded);
