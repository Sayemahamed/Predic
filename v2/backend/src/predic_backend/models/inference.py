import asyncio
from typing import List, Dict, Any, AsyncGenerator, Union
import torch
from transformers import TextIteratorStreamer
from threading import Thread

from .manager import model_manager


class InferenceEngine:
    
    async def generate_completion(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.2,
        stop_sequences: List[str] = None,
        stream: bool = False,
    ) -> Union[AsyncGenerator[str, None], str]:
        """Generate code completion."""
        
        # Ensure model is loaded
        if not model_manager.get_loaded_model(model_id):
            await model_manager.load_model(model_id)
            
        model_data = model_manager.get_loaded_model(model_id)
        if not model_data:
            raise ValueError(f"Failed to load model: {model_id}")
            
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        if stop_sequences:
            # Convert stop sequences to token IDs
            stop_ids = []
            for seq in stop_sequences:
                tokens = tokenizer.encode(seq, add_special_tokens=False)
                if tokens:
                    stop_ids.extend(tokens)
            if stop_ids:
                gen_kwargs["eos_token_id"] = stop_ids
                
        if stream:
            # Return async generator for streaming
            return self._stream_generation(model, tokenizer, inputs, gen_kwargs)
        else:
            # Non-streaming generation
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    model.generate,
                    inputs["input_ids"],
                    **gen_kwargs,
                )
                
            # Decode output
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            
            return generated_text
    
    async def _stream_generation(
        self,
        model,
        tokenizer,
        inputs,
        gen_kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generation tokens."""
        streamer = TextIteratorStreamer(
            tokenizer, 
            skip_prompt=True,
            skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer
        
        # Run generation in a separate thread
        thread = Thread(
            target=model.generate,
            args=(inputs["input_ids"],),
            kwargs=gen_kwargs,
        )
        thread.start()
        
        # Yield tokens as they're generated
        for token in streamer:
            yield token
            
        thread.join()
            
    async def chat(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Union[AsyncGenerator[str, None], str]:
        """Handle chat conversation."""
        
        # Ensure model is loaded
        if not model_manager.get_loaded_model(model_id):
            await model_manager.load_model(model_id)
            
        model_data = model_manager.get_loaded_model(model_id)
        if not model_data:
            raise ValueError(f"Failed to load model: {model_id}")
            
        tokenizer = model_data["tokenizer"]
        
        # Format messages into a prompt
        prompt = self._format_chat_prompt(messages, model_id)
        
        # Generate response
        response = await self.generate_completion(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        
        return response
        
    def _format_chat_prompt(
        self, messages: List[Dict[str, str]], model_id: str
    ) -> str:
        
        # Different models have different chat formats
        if "codellama" in model_id.lower():
            # CodeLlama format
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"<<SYS>>\n{msg['content']}\n<</SYS>>\n\n"
                elif msg["role"] == "user":
                    prompt += f"[INST] {msg['content']} [/INST]\n"
                elif msg["role"] == "assistant":
                    prompt += f"{msg['content']}\n"
            return prompt
            
        elif "deepseek" in model_id.lower():
            # DeepSeek format
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"### Instruction:\n{msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"### Response:\n{msg['content']}\n\n"
            prompt += "### Response:\n"
            return prompt
        
        else:
            # Generic format
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
            prompt += "Assistant: "
            return prompt


# Global inference engine instance
inference_engine = InferenceEngine()