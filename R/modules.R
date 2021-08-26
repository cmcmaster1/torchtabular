continuous_embedding <- torch::nn_module(
  "continuous_embedding",
  initialize = function(intermediate_dims, embedding_dim) {
    self$layers <- torch::nn_sequential(
      torch::nn_linear(1, intermediate_dims),
      torch::nn_relu(),
      torch::nn_linear(intermediate_dims, embedding_dim)
    )
  },
  forward = function(x) {
    if (length(x$shape) == 1){
      x = x$view(c(x$size(1), -1))
    }
    self$layers(x)
  })


attention <- torch::nn_module(
  "attention",
  initialize = function(input_dim, dim_head, num_heads, softmax_mod = 1, attention_type = "softmax") {
    self$dim_head <- dim_head
    self$num_heads <- num_heads
    self$inner_dim <- dim_head *num_heads

    self$qkv_proj <- torch::nn_linear(input_dim, 3*self$inner_dim)
    self$o_proj <- torch::nn_linear(self$inner_dim, input_dim)

    self$softmax_mod <- softmax_mod
    self$attention_type <- attention_type

    if (self$attention_type == "signed"){
      self$k_relu <- torch::nn_relu()
    }
  },
  forward = function(x, return_attention=FALSE) {
    batch_size <- x$shape[1]
    seq_length <- x$shape[2]
    embed_dim <- x$shape[3]

    qkv <- self$qkv_proj(x)
    qkv <- qkv$reshape(c(batch_size, seq_length, self$num_heads, 3*self$dim_head))
    qkv <- qkv$permute(c(1, 3, 2, 4))
    q_k_v <- qkv$chunk(3, dim=-1)
    q <- q_k_v[[1]]
    k <- q_k_v[[2]]
    v <- q_k_v[[3]]

    qs <- length(q$shape)
    d_k <- q$shape[qs]
    attn_logits <- torch::torch_matmul(q, k$transpose(-2, -1))
    attn_logits <- attn_logits / sqrt(d_k)

    if (self$attention_type == "signed"){
      v <- self$k_relu(v)
      attention <- torch::nnf_normalize(attn_logits, p = 1, dim = -1)
    } else {
      attention <- torch::nnf_softmax(self$softmax_mod * attn_logits, dim=-1)
    }

    attention <- torch::nnf_softmax(self$softmax_mod * attn_logits, dim=-1)

    values <- torch::torch_matmul(attention, v)

    values <- values$permute(c(1, 3, 2, 4))
    values <- values$reshape(c(batch_size, seq_length, self$inner_dim))
    o <- self$o_proj(values)

    if (return_attention == TRUE){
      list(o, attention)
    } else{
      o
    }
  }
)


signed_attention <- torch::nn_module(
  "signed_attention",
  initialize = function(input_dim, dim_head, num_heads, softmax_mod = 1) {
    self$dim_head <- dim_head
    self$num_heads <- num_heads
    self$inner_dim <- dim_head *num_heads

    self$qkv_proj <- torch::nn_linear(input_dim, 3*self$inner_dim)
    self$o_proj <- torch::nn_linear(self$inner_dim, input_dim)
    self$k_relu <- torch::nn_relu()

    self$softmax_mod <- softmax_mod
  },
  forward = function(x, return_attention=FALSE) {
    batch_size <- x$shape[1]
    seq_length <- x$shape[2]
    embed_dim <- x$shape[3]

    qkv <- self$qkv_proj(x)
    qkv <- qkv$reshape(c(batch_size, seq_length, self$num_heads, 3*self$dim_head))
    qkv <- qkv$permute(c(1, 3, 2, 4))
    q_k_v <- qkv$chunk(3, dim=-1)
    q <- q_k_v[[1]]
    k <- q_k_v[[2]]
    v <- q_k_v[[3]]
    v <- self$k_relu(v)

    qs <- length(q$shape)
    d_k <- q$shape[qs]
    attn_logits <- torch::torch_matmul(q, k$transpose(-2, -1))
    attn_logits <- attn_logits / sqrt(d_k)

    attention <- torch::torch_sign(attn_logits) *
      (torch::nnf_softmax(-self$softmax_mod * attn_logits, -1) +
         torch::nnf_softmax(self$softmax_mod * attn_logits, -1))/2

    # attention <- torch::nnf_normalize(attn_logits, p = 1, dim = -1)

    values <- torch::torch_matmul(attention, v)

    values <- values$permute(c(1, 3, 2, 4))
    values <- values$reshape(c(batch_size, seq_length, self$inner_dim))
    o <- self$o_proj(values)

    if (return_attention == TRUE){
      list(o, attention)
    } else{
      o
    }
  }
)

geglu <- torch::nn_module(
  "geglu",
  initialize = function(){
  },
  forward = function(x){
    x_chunk <- x$chunk(2, dim=-1)
    torch::torch_mul(x_chunk[[1]], torch::nnf_gelu(x_chunk[[2]]))
  }
)


ff <- torch::nn_module(
  "ff",
  initialize = function(dim, dropout = 0.1, mult = 4){
    self$main <- torch::nn_sequential(
      torch::nn_linear(dim, dim * mult * 2),
      geglu(),
      torch::nn_dropout(dropout),
      torch::nn_linear(dim * mult, dim)
    )
  },
  forward = function(x){
    self$main(x)
  }
)



tabular_mlp <- torch::nn_module(
  "tabular_mlp",
  initialize = function(dims, final_layer = NULL, mlp_dropout = 0.1){
    dim_pairs <- into_pairs(dims)
    layers_fn <- function(x) {
      torch::nn_linear(x[1],x[2])
    }
    layers <- lapply(dim_pairs, layers_fn)

    self$mlp <- torch::nn_sequential()
    mapply(function(x, y) self$mlp$add_module(name = x, module = y), names(layers), layers)

    final_layer <- final_layer %||% torch::nn_identity()
    self$mlp$add_module("final_layer", final_layer)

  },
  forward = function(x)
    self$mlp(x)
)
