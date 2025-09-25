import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.utils import logging


class ReaCompleteConfig(PretrainedConfig):
    model_type = "ReaComplete"
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=True,
        **kwargs,
    ) -> None:
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * self.n_embd
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )


# YOUR ORIGINAL, CORRECT ATTENTION MODULE IS PRESERVED
class ReaCompleteAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        max_positions = config.n_positions
        self.register_buffer(
            name="bias",
            tensor=torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.layer_idx = layer_idx

    def _split_heads(self, tensor, num_heads, attention_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attention_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)

    def forward(
        self, hidden_states, layer_past=None, attention_mask=None, use_cache=False
    ):
        query, key, value = self.c_attn(hidden_states).split(self.embed_dim, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ].to(torch.bool)

        mask_value = torch.finfo(query.dtype).min
        # This combines the causal mask with the padding mask
        attn_mask = torch.where(causal_mask, 0.0, mask_value)
        if attention_mask is not None:
            attn_mask = attn_mask + attention_mask

        # is_causal=False is correct here because we have manually created the combined causal and padding mask
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present


# YOUR RESEARCH COMPONENT - UNCHANGED
class ReaCompleteFeedForward(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.network = nn.Sequential(
            nn.Linear(embed_dim, intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(intermediate_size, intermediate_size),
            nn.Dropout(config.resid_pdrop),
            nn.GELU(approximate="tanh"),
            nn.Linear(intermediate_size, embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, hidden_states):
        return self.network(hidden_states)


class ReaCompleteDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = ReaCompleteAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = ReaCompleteFeedForward(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_output, present_key_value = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ReaCompletePreTrainedModel(PreTrainedModel):
    config_class = ReaCompleteConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ReaCompleteDecoderLayer"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ReaCompleteModel):
            module.gradient_checkpointing = value


class ReaCompleteModel(ReaCompletePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                ReaCompleteDecoderLayer(config, layer_idx=i)
                for i in range(config.n_layer)
            ]
        )
        self.ln_f = nn.RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        presents = [] if use_cache else None
        all_hidden_states = [] if output_hidden_states else None

        for block, layer_past in zip(self.h, past_key_values):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                )

            hidden_states = outputs[0]
            if use_cache:
                presents.append(outputs[1])

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if presents is not None:
            presents = tuple(presents)
        if all_hidden_states is not None:
            all_hidden_states = tuple(all_hidden_states)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class ReaCompleteHeadModel(ReaCompletePreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.2.weight",
    ]
    _tied_weights_keys = ["lm_head.2.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ReaCompleteModel(config)
        # YOUR RESEARCH COMPONENT - UNCHANGED
        self.lm_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.n_embd, config.vocab_size, bias=False),
        )
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head[-1]

    def set_output_embeddings(self, new_embeddings):
        self.lm_head[-1] = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


# Register the custom model with the AutoClasses
AutoConfig.register("ReaComplete", ReaCompleteConfig)
AutoModelForCausalLM.register(ReaCompleteConfig, ReaCompleteHeadModel)


if __name__ == "__main__":
    tokenizer_name = "openai-community/gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Could not load {tokenizer_name} tokenizer: {e}. Trying 'gpt2'...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    config = ReaCompleteConfig(
        # vocab_size=tokenizer.vocab_size,
        # n_positions=128,
        # n_embd=128,
        # n_layer=3,
        # n_head=4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # tie_word_embeddings=True,
    )

    print("Instantiating model from config...")
    model = ReaCompleteHeadModel(config)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    print("\n--- Testing Save/Load ---")
    save_directory = "./my_custom_gpt2_model_full"
    model.save_pretrained(save_directory)
    print(f"Model saved to {save_directory}")

    loaded_model = AutoModelForCausalLM.from_pretrained(save_directory)
    loaded_model.eval()
    print("Model loaded successfully using AutoModelForCausalLM.")
    assert torch.allclose(model.lm_head[2].weight, loaded_model.lm_head[2].weight)
    print("Saved and loaded model weights match (tied weights check).")

    print("\n--- Testing Forward Pass ---")
    input_text = "Hello, my name is"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids.clone())
        loss = outputs.loss
        logits = outputs.logits
    print(f"Input: '{input_text}'")
    print("Input IDs shape:", input_ids.shape)
    print("Logits shape:", logits.shape)
    print("Calculated loss:", loss.item() if loss is not None else "N/A")
    assert logits.shape == (input_ids.shape[0], input_ids.shape[1], config.vocab_size)

    print("\n--- Testing Generation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt = "Once upon a time"
    input_ids_gen = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(
        f"Prompt for generation: '{prompt}' (Input ID length: {input_ids_gen.shape[1]})"
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids_gen,
            max_length=30,
            num_return_sequences=1,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    generated_text = tokenizer.decode(
        generated_ids[0].tolist(), skip_special_tokens=True
    )
    print(f"Generated text: '{generated_text}'")
    assert len(generated_ids[0]) <= 30

    print("\n--- Testing Pipeline ---")
    from transformers import pipeline

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=model.device
    )
    pipeline_output = pipe(
        "In a galaxy far, far away", max_length=30, num_return_sequences=1
    )
    print(f"Pipeline generated text: '{pipeline_output[0]['generated_text']}'")

    print("\n--- Testing Trainer Compatibility (Minimal) ---")
    dummy_texts = ["This is a test sentence " + str(i) for i in range(10)]

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.encodings = [
                tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
                for text in texts
            ]

        def __getitem__(self, idx):
            item = {key: val.squeeze(0) for key, val in self.encodings[idx].items()}
            item["labels"] = item["input_ids"].clone()
            return item

        def __len__(self):
            return len(self.encodings)

    train_dataset = DummyDataset(dummy_texts, tokenizer, max_length=config.n_positions)
    training_args = TrainingArguments(
        output_dir="./my_custom_gpt2_trainer_results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    try:
        print("Starting dummy training step...")
        trainer.train()
        print("Dummy training step completed.")
    except Exception as e:
        print(f"Error during dummy training: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Hugging Face Compatibility Test Done ---")
