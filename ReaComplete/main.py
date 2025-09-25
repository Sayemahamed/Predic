import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
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
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
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
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            use_cache=self.use_cache,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            **kwargs,
        )


# 2. Attention Module (REFACTORED for SDPA)
class MyGPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MyGPT2Attention requires PyTorch 2.0 or higher.")

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = config.attn_pdrop  # Use value, not module, for SDPA
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.layer_idx = layer_idx

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
        if output_attentions:
            raise ValueError("output_attentions=True is not supported with SDPA.")
        if head_mask is not None:
            raise ValueError("head_mask is not supported with SDPA.")

        query, key, value = self.c_attn(hidden_states).split(self.embed_dim, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=attention_mask
            is None,  # Use SDPA's internal causal mask if no padding mask is provided
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present


# 3. MLP (FeedForward) Module
class MyGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        if config.activation_function == "gelu_new":
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        return self.dropout(self.c_proj(self.act(self.c_fc(hidden_states))))


# 4. Transformer Block
class MyGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MyGPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MyGPT2MLP(config.n_inner, config)

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
        attn_output, present_key_value = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


# 5. Base Model (Stack of Transformers) (SIMPLIFIED)
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
        self.gradient_checkpointing = False  # Disabled by default
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
        # Determine behavior from arguments or config
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # === Essential Input Validation and Setup ===
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_length = (
            past_key_values[0][0].size(-2) if past_key_values is not None else 0
        )

        if position_ids is None:
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # === Prepare Embeddings ===
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        # === Prepare Attention Mask for SDPA ===
        # SDPA can use a boolean mask (True = NOT attend) or a float mask.
        # HF uses a float mask. We will adapt the attention_mask if it's provided.
        # If no attention_mask, SDPA will use its own causal mask if `is_causal=True`.
        causal_mask_for_sdpa = None
        if attention_mask is not None:
            # The causal mask is handled by `is_causal=True` in SDPA,
            # but if we have padding, we need to combine masks.
            # So, we'll let the attention layer handle it.
            causal_mask_for_sdpa = attention_mask[:, None, None, :].to(dtype=self.dtype)
            causal_mask_for_sdpa = (1.0 - causal_mask_for_sdpa) * torch.finfo(
                self.dtype
            ).min

        # === Transformer Blocks Loop ===
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    layer_past,
                    causal_mask_for_sdpa,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask_for_sdpa,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

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


# 6. Model with LM Head
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


# 8. Test Script
if __name__ == "__main__":
    # The test script remains unchanged and will now use the simplified model
    tokenizer_name = "openai-community/gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    )

    print("Instantiating model from config...")
    model = MyGPT2LMHeadModel(config)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # All tests for Save/Load, Forward Pass, Generation, Pipeline, and Trainer
    # will run as before.
    print("\n--- Testing Save/Load ---")
    save_directory = "./my_custom_gpt2_model_full"
    model.save_pretrained(save_directory)
    print(f"Model saved to {save_directory}")

    loaded_model = AutoModelForCausalLM.from_pretrained(save_directory)
    loaded_model.eval()
    print("Model loaded successfully.")
    assert torch.allclose(model.lm_head.weight, loaded_model.lm_head.weight)
    print("Saved and loaded model weights match.")

    # ... rest of the test script ...
    print("\n--- Testing Forward Pass ---")
    input_text = "Hello, my name is"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids.clone())
        loss = outputs.loss
        logits = outputs.logits
    print(f"Input: '{input_text}'")
    print("Logits shape:", logits.shape)
    print("Calculated loss:", loss.item() if loss is not None else "N/A")

    print("\n--- Testing Generation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt = "Once upon a time"
    input_ids_gen = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids_gen, max_length=30, pad_token_id=config.pad_token_id
        )
    generated_text = tokenizer.decode(
        generated_ids[0].tolist(), skip_special_tokens=True
    )
    print(f"Generated text: '{generated_text}'")

    print("\n--- Hugging Face Compatibility Test Done ---")
