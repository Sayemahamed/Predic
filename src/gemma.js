import { pipeline } from "@huggingface/transformers";

// Create a text generation pipeline
const generator = await pipeline(
  "text-generation",
  "onnx-community/gemma-3-1b-it-ONNX-GQA",
  { dtype: "q4" }
);

// Define the list of messages
const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Write a simple react button" },
];

// Generate a response
const output = await generator(messages, {
  max_new_tokens: 512,
  do_sample: false,
});
console.log(output[0].generated_text.at(-1).content);
