import torch
from torch import nn
from torch.nn import functional as F
from transformers.configuration_utils import PretrainedConfig


class MyCustomTransformerConfig(PretrainedConfig):
    model_type = "my_custom_transformer"  # Important for AutoClass registration

    def __init__(
        self,
        vocab_size: int = 30522,
        embedding_dim: int = 768,
        context_length: int = 512,
        number_of_decoders: int = 4,
        number_of_heads: int = 8,
        dropout: float = 0.1,
        is_decoder_flag: bool = True,  # Used for MHA's is_causal flag
        expansion_factor: int = 4,
        output_bias: bool = True,  # Bias for MHA output projection and LM head layers
        qkv_bias: bool = True,  # Bias for MHA QKV projections
        ffn_bias: bool = True,  # Bias for FFN layers
        initializer_range: float = 0.02,
        torch_dtype: str = "float16",  # Can be "float32", "float16", "bfloat16"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.number_of_decoders = number_of_decoders
        self.number_of_heads = number_of_heads
        self.dropout = dropout
        self.is_decoder_flag = is_decoder_flag
        self.expansion_factor = expansion_factor
        self.output_bias = output_bias
        self.qkv_bias = qkv_bias
        self.ffn_bias = ffn_bias
        self.initializer_range = initializer_range

        # Convert string dtype to actual torch.dtype
        if isinstance(torch_dtype, str):
            try:
                self.torch_dtype = getattr(torch, torch_dtype)
            except AttributeError:
                raise ValueError(f"Invalid torch_dtype: {torch_dtype}")
        else:
            self.torch_dtype = torch_dtype  # Assume it's already a torch.dtype object

        if embedding_dim % number_of_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by number_of_heads ({number_of_heads})"
            )


class MultiHeadAttention(nn.Module):
    def __init__(self, config: MyCustomTransformerConfig) -> None:
        """
        Multi-Head Attention module.

        Args:
            config: Configuration object for the transformer.
        """
        super().__init__()

        self.embedding_dim = config.embedding_dim
        self.number_of_heads = config.number_of_heads
        self.head_dim = (
            config.embedding_dim // config.number_of_heads
        )  # Checked in config

        self.attention_dropout_p = config.dropout  # For F.scaled_dot_product_attention
        self.context_length = config.context_length  # For input validation
        self.is_causal_mha = (
            config.is_decoder_flag
        )  # For F.scaled_dot_product_attention's is_causal

        self.query_key_value = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.embedding_dim * 3,
            bias=config.qkv_bias,
            dtype=config.torch_dtype,
        )
        self.output = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.embedding_dim,
            bias=config.output_bias,
            dtype=config.torch_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, current_embedding_dim = x.shape
        if current_embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Input embedding_dim ({current_embedding_dim}) doesn't match model embedding_dim ({self.embedding_dim})"
            )
        if sequence_length > self.context_length:
            raise ValueError(
                f"Input sequence_length ({sequence_length}) exceeds model context_length ({self.context_length})"
            )

        qkv = self.query_key_value(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size, sequence_length, self.number_of_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, sequence_length, self.number_of_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, sequence_length, self.number_of_heads, self.head_dim
        ).transpose(1, 2)

        # F.scaled_dot_product_attention handles dropout internally if dropout_p > 0 and model is in training mode.
        # However, it's safer to explicitly pass 0.0 if not training, as per documentation for some versions/backends.
        current_dropout_p = self.attention_dropout_p if self.training else 0.0

        attention_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,  # Handles causal masking if is_causal=True
            dropout_p=current_dropout_p,
            is_causal=self.is_causal_mha,
        )
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.embedding_dim)
        )
        return self.output(attention_output)


class FeedForwardNetwork(nn.Module):
    """
    A simple FeedForward Network module.
    Structure: Linear -> GELU -> Dropout -> Linear.
    """

    def __init__(self, config: MyCustomTransformerConfig) -> None:
        super().__init__()
        hidden_dim = config.embedding_dim * config.expansion_factor

        self.network = nn.Sequential(
            nn.Linear(
                in_features=config.embedding_dim,
                out_features=hidden_dim,
                dtype=config.torch_dtype,
                bias=config.ffn_bias,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                in_features=hidden_dim,
                out_features=config.embedding_dim,
                dtype=config.torch_dtype,
                bias=config.ffn_bias,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, config: MyCustomTransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward_network = FeedForwardNetwork(config)

        self.layer_norm1 = nn.LayerNorm(
            normalized_shape=config.embedding_dim, dtype=config.torch_dtype
        )
        self.layer_norm2 = nn.LayerNorm(
            normalized_shape=config.embedding_dim, dtype=config.torch_dtype
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is assumed to be of config.torch_dtype
        attn_output = self.self_attention(self.layer_norm1(x))
        x = x + self.dropout(attn_output)  # Residual connection

        ffn_output = self.feed_forward_network(self.layer_norm2(x))
        x = x + self.dropout(ffn_output)  # Residual connection
        return x


class Model(
    nn.Module
):  # Consider inheriting from transformers.PreTrainedModel for full HF integration
    def __init__(self, config: MyCustomTransformerConfig) -> None:
        super().__init__()
        self.config = config  # Store config

        # Embeddings are often kept in float32 for numerical stability, then cast.
        # Here, they will initialize as float32 by default.
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=config.context_length, embedding_dim=config.embedding_dim
        )

        self.decoders = nn.ModuleList(
            [Decoder(config) for _ in range(config.number_of_decoders)]
        )

        self.layer_norm = nn.LayerNorm(
            normalized_shape=config.embedding_dim, dtype=config.torch_dtype
        )

        # LM Head (structure from original code)
        lm_head_hidden_dim = config.embedding_dim * config.expansion_factor
        self.lm_head = nn.Sequential(
            nn.Linear(
                in_features=config.embedding_dim,
                out_features=lm_head_hidden_dim,
                bias=config.output_bias,  # Using output_bias for LM head layers
                dtype=config.torch_dtype,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=lm_head_hidden_dim,
                out_features=lm_head_hidden_dim,
                bias=config.output_bias,
                dtype=config.torch_dtype,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=lm_head_hidden_dim,
                out_features=config.vocab_size,
                bias=config.output_bias,
                dtype=config.torch_dtype,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=lm_head_hidden_dim,
                out_features=config.vocab_size,
                bias=config.output_bias,
                dtype=config.torch_dtype,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=lm_head_hidden_dim,
                out_features=config.vocab_size,
                bias=config.output_bias,
                dtype=config.torch_dtype,
            ),
        )
        # If this model were a transformers.PreTrainedModel, you'd typically call:
        # self.post_init()
        # to handle weight initialization according to config.initializer_range, etc.

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Input sequence_length ({sequence_length}) exceeds model's configured context_length ({self.config.context_length})"
            )

        token_embeds = self.token_embedding(input_ids)  # (B, S, D)

        # Create position_ids on the fly: (S) -> (1, S)
        position_ids = torch.arange(
            0, sequence_length, device=input_ids.device
        ).unsqueeze(0)
        pos_embeds = self.positional_embedding(
            position_ids
        )  # (1, S, D), broadcasts with token_embeds

        # Sum embeddings and cast to the model's working dtype
        x = (token_embeds + pos_embeds).to(self.config.torch_dtype)

        for decoder_layer in self.decoders:
            x = decoder_layer(x)

        x = self.layer_norm(x)
        logits = self.lm_head(x)

        # Optionally, cast logits to float32 if mixed precision is used and loss calculation requires float32
        # For example, if config.torch_dtype is torch.float16 or torch.bfloat16:
        # if logits.dtype != torch.float32:
        #     logits = logits.to(torch.float32)

        return logits
