softmax_attention <- torch::nn_module(
  "softmax_attention",
  initialize = function(input_dim, dim_head, num_heads, softmax_mod = 1, attention_type = "softmax") {
    self$dim_head <- dim_head
    self$num_heads <- num_heads
    self$inner_dim <- dim_head *num_heads

    self$qkv_proj <- torch::nn_linear(input_dim, 3*self$inner_dim)
    self$o_proj <- torch::nn_linear(self$inner_dim, input_dim)

    self$softmax_mod <- softmax_mod
    self$attention_type <- attention_type

  },
  forward = function(x, return_attention=FALSE) {
    batch_size <- x$shape[1]
    seq_length <- x$shape[2]
    embed_dim <- x$shape[3]

    qkv <- self$qkv_proj(x)
    qkv <- qkv$chunk(3, dim=-1)
    qkv <- lapply(qkv,
                  function(x) {
                    x$reshape(c(batch_size, seq_length, self$num_heads, self$dim_head))$permute(c(1, 3, 2, 4))
                  }
    )

    q <- qkv[[1]]
    k <- qkv[[2]]
    v <- qkv[[3]]

    qs <- length(q$shape)
    d_k <- q$shape[qs]
    attn_logits <- torch::torch_matmul(q, k$transpose(-2, -1))
    attn_logits <- attn_logits / sqrt(d_k)

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
    self$v_relu <- torch::nn_relu()

    self$softmax_mod <- softmax_mod
  },
  forward = function(x, return_attention=FALSE) {
    batch_size <- x$shape[1]
    seq_length <- x$shape[2]
    embed_dim <- x$shape[3]

    qkv <- self$qkv_proj(x)
    qkv <- qkv$chunk(3, dim=-1)
    qkv <- lapply(qkv,
                  function(x) {
                    x$reshape(c(batch_size, seq_length, self$num_heads, self$dim_head))$permute(c(1, 3, 2, 4))
                  }
    )

    q <- qkv[[1]]
    k <- qkv[[2]]
    v <- qkv[[3]]

    v <- self$v_relu(v)

    qs <- length(q$shape)
    d_k <- q$shape[qs]
    attn_logits <- torch::torch_matmul(q, k$transpose(-2, -1))
    attn_logits <- attn_logits / sqrt(d_k)

    attention <- torch::nnf_normalize(attn_logits, p = 1, dim = -1)

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

fast_attention <- torch::nn_module(
  "fast_attention",
  initialize = function(input_dim, dim_head, num_heads) {
    self$dim_head <- dim_head
    self$num_heads <- num_heads
    self$inner_dim <- dim_head *num_heads
    self$scale <- dim_head ^ -0.5

    self$qkv_proj <- torch::nn_linear(input_dim, 3*self$inner_dim, bias = FALSE)
    self$to_q_attn_logits <- torch::nn_linear(self$dim_head, 1, bias = FALSE)
    self$to_k_attn_logits <- torch::nn_linear(self$dim_head, 1, bias = FALSE)

    self$to_r <- torch::nn_linear(self$dim_head, self$dim_head)
    self$to_out <- torch::nn_linear(self$inner_dim, input_dim)
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

    q_aggr <- q
    k_aggr <- k
    v_aggr <- v

    q_attn_logits <- self$to_q_attn_logits(q)$squeeze(-1) * self$scale
    q_attn <- q_attn_logits$softmax(dim = -1)

    global_q <- torch::torch_einsum('b h n, b h n d -> b h d', list(q_attn, q_aggr))
    global_q$unsqueeze_(-2)

    k <- k * global_q

    k_attn_logits <- self$to_k_attn_logits(k)$squeeze(-1) * self$scale
    k_attn <- k_attn_logits$softmax(dim = -1)

    global_k <- torch::torch_einsum('b h n, b h n d -> b h d', list(k_attn, k_aggr))
    global_k$unsqueeze_(-2)

    u <- v_aggr * global_k
    r <- self$to_r(u)

    r <- r + q

    r <- r$permute(c(1, 3, 2, 4))
    r <- r$reshape(c(batch_size, seq_length, self$inner_dim))
    out <- self$to_out(r)

    if (return_attention == TRUE){
      list(out, k_attn)
    } else{
      out
    }
  }
)

sparsemax_attention <- torch::nn_module(
  "sparsemax_attention",
  initialize = function(input_dim, dim_head, num_heads) {
    self$dim_head <- dim_head
    self$num_heads <- num_heads
    self$inner_dim <- dim_head *num_heads

    self$qkv_proj <- torch::nn_linear(input_dim, 3*self$inner_dim)
    self$o_proj <- torch::nn_linear(self$inner_dim, input_dim)
    self$sparsemax <- sparsemax(dim = -1)

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

    attention <- self$sparsemax(attn_logits)

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

attention <- function(attention_type = "softmax", input_dim, dim_head, num_heads, ...){
  if(attention_type == "softmax"){
    softmax_attention(input_dim, dim_head, num_heads, ...)
  } else if(attention_type == "sparsemax"){
    sparsemax_attention(input_dim, dim_head, num_heads)
  } else if(attention_type == "fast"){
    fast_attention(input_dim, dim_head, num_heads)
  } else if(attention_type == "signed"){
    signed_attention(input_dim, dim_head, num_heads, ...)
  } else{
    stop("Unknown attention type")
  }
}
