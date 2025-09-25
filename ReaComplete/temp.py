import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM  # For Trainer test
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin  # For Generation
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


# 1. Configuration Class
class MyGPT2Config(PretrainedConfig):
    model_type = "my_gpt2"
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
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=True,
        use_cache=True,  # ADDED
        output_attentions=False,  # ADDED
        output_hidden_states=False,  # ADDED
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * self.n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache  # ASSIGNED
        self.output_attentions = output_attentions  # ASSIGNED
        self.output_hidden_states = output_hidden_states  # ASSIGNED

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            use_cache=self.use_cache,  # PASSED TO SUPER
            output_attentions=self.output_attentions,  # PASSED TO SUPER
            output_hidden_states=self.output_hidden_states,  # PASSED TO SUPER
            **kwargs,
        )


# 2. Attention Module
class MyGPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        max_positions = config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.layer_idx = layer_idx

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        query, key, value = self.c_attn(hidden_states).split(self.embed_dim, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


# 3. MLP (FeedForward) Module
class MyGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        if config.activation_function == "gelu_new":
            self.act = nn.GELU(approximate="tanh")
        elif config.activation_function == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 4. Transformer Block
class MyGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MyGPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MyGPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


# 5. Base Model (Stack of Transformers)
class MyGPT2PreTrainedModel(PreTrainedModel):
    config_class = MyGPT2Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MyGPT2Block"]

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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MyGPT2Model):
            module.gradient_checkpointing = value


class MyGPT2Model(MyGPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [MyGPT2Block(config, layer_idx=i) for i in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
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
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )  # This should now work
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
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
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i] if head_mask is not None else None,
                    use_reentrant=False,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# 6. Model with LM Head - Inherit from GenerationMixin
class MyGPT2LMHeadModel(MyGPT2PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.weight",
    ]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = MyGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            output_embeddings = self.get_output_embeddings()
            if (
                output_embeddings is not None
                and self.transformer.get_input_embeddings() is not None
            ):
                self._tie_or_clone_weights(
                    output_embeddings, self.transformer.get_input_embeddings()
                )
            elif (
                output_embeddings is None
                or self.transformer.get_input_embeddings() is None
            ):
                logger.warning(
                    "Tie_word_embeddings set to True but an embedding matrix is None for MyGPT2LMHeadModel."
                )

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
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
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
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
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


# 7. Register with AutoClass
AutoConfig.register("my_gpt2", MyGPT2Config)
AutoModelForCausalLM.register(MyGPT2Config, MyGPT2LMHeadModel)


if __name__ == "__main__":
    tokenizer_name = "openai-community/gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Could not load {tokenizer_name} tokenizer: {e}. Trying 'gpt2'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e_fallback:
            print(f"Could not load 'gpt2' tokenizer either: {e_fallback}")
            print(
                "Proceeding with dummy tokenizer logic, generation will be meaningless."
            )

            class DummyTokenizer:
                def __init__(self, vocab_size, bos_token, eos_token, pad_token):
                    self.vocab = {
                        bos_token: 0,
                        eos_token: 1,
                        pad_token: 2,
                        "a": 3,
                        "b": 4,
                        "c": 5,
                    }
                    self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
                    self.vocab_size = len(self.vocab)
                    self.bos_token_id = self.vocab[bos_token]
                    self.eos_token_id = self.vocab[eos_token]
                    self.pad_token_id = self.vocab[pad_token]
                    self.bos_token = bos_token
                    self.eos_token = eos_token
                    self.pad_token = pad_token

                def encode(self, text, return_tensors=None, **kwargs):
                    ids = [self.vocab.get(c, self.vocab_size - 1) for c in text[:10]]
                    if return_tensors == "pt":
                        return torch.tensor([ids])
                    return [ids]

                def decode(self, token_ids, skip_special_tokens=False):
                    tokens = []
                    for tid in token_ids:
                        token = self.ids_to_tokens.get(tid, "[UNK]")
                        if skip_special_tokens and tid in [
                            self.bos_token_id,
                            self.eos_token_id,
                            self.pad_token_id,
                        ]:
                            continue
                        tokens.append(token)
                    return "".join(tokens)

                def __call__(self, text, return_tensors=None, **kwargs):
                    encoded = self.encode(text, return_tensors=return_tensors, **kwargs)
                    if isinstance(encoded, torch.Tensor):
                        return {
                            "input_ids": encoded,
                            "attention_mask": torch.ones_like(encoded),
                        }
                    return {"input_ids": encoded}

            tokenizer = DummyTokenizer(
                50, "<|endoftext|>", "<|endoftext|>", "<|endoftext|>"
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Also ensure tokenizer.pad_token_id is set if pad_token was None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    config = MyGPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_embd=128,
        n_layer=3,
        n_head=4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
        # Default use_cache, output_attentions, output_hidden_states are fine
        # as they are set in MyGPT2Config's __init__
    )

    print("Instantiating model from config...")
    model = MyGPT2LMHeadModel(config)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    print("\n--- Testing Save/Load ---")
    save_directory = "./my_custom_gpt2_model_full"
    model.save_pretrained(save_directory)
    print(f"Model saved to {save_directory}")

    loaded_model = AutoModelForCausalLM.from_pretrained(save_directory)
    loaded_model.eval()
    print("Model loaded successfully using AutoModelForCausalLM.")
    assert torch.allclose(model.lm_head.weight, loaded_model.lm_head.weight)
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
    print("Calculated loss:", loss.item() if loss is not None else "N/A (no labels)")
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
            pad_token_id=config.pad_token_id,  # Use config's pad_token_id
            eos_token_id=config.eos_token_id,  # Use config's eos_token_id
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
            self.encodings = []
            for text in texts:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
                self.encodings.append(
                    {
                        "input_ids": encoding.input_ids.squeeze(0),
                        "attention_mask": encoding.attention_mask.squeeze(0),
                        "labels": encoding.input_ids.squeeze(0).clone(),
                    }
                )

        def __getitem__(self, idx):
            return self.encodings[idx]

        def __len__(self):
            return len(self.encodings)

    train_dataset = DummyDataset(dummy_texts, tokenizer, max_length=config.n_positions)

    training_args = TrainingArguments(
        output_dir="./my_custom_gpt2_trainer_results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    try:
        print("Starting dummy training step...")
        trainer.train()
        print("Dummy training step completed.")
    except Exception as e:
        print(f"Error during dummy training: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- Hugging Face Compatibility Test Done ---")
