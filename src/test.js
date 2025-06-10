import { AutoTokenizer, BartTokenizer } from "@huggingface/transformers";

// Load tokenizer for a gated repository.
const tokenizer = await AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B");
const custom_tokenizer = await new BartTokenizer("./tokenizer.json");
// Encode text.
const text = "Hello world!";
const encoded = tokenizer.encode(text);
console.log(encoded);
const decoded = tokenizer.decode(encoded);
console.log(decoded);

const custom_encoded = custom_tokenizer.encode(text);
console.log(custom_encoded);
const custom_decoded = custom_tokenizer.decode(custom_encoded);
console.log(custom_decoded);
