continuous_embedding <- torch::nn_module(
  "continuous_embedding",
  initialize = function(
    intermediate_dims,
    embedding_dim) {
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
  initialize = function(input_dim, dim_head, num_heads, softmax_mod = 1) {
    self$dim_head <- dim_head
    self$num_heads <- num_heads
    self$inner_dim <- dim_head *num_heads

    self$qkv_proj <- torch::nn_linear(input_dim, 3*self$inner_dim)
    self$o_proj <- torch::nn_linear(self$inner_dim, input_dim)

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


tabular_transformer <- torch::nn_module(
  "tabular_transformer",
  initialize = function(
    dim,
    cols,
    depth,
    heads_selfattn,
    heads_intersample,
    dim_heads_selfattn,
    dim_heads_intersample,
    attn_dropout,
    ff_dropout,
    softmax_mod,
    intersample = TRUE)
  {
    self$intersample <- intersample
    self$layers <- torch::nn_module_list()

    if (intersample){
      for (i in 1:depth){
        self$layers$append(
          torch::nn_module_list(
            list(
              attention(dim, heads_selfattn, dim_heads_selfattn, softmax_mod),
              torch::nn_dropout(p = attn_dropout),
              torch::nn_layer_norm(dim),
              ff(dim, dropout = ff_dropout),
              torch::nn_layer_norm(dim),
              attention(dim * cols, heads_intersample, dim_heads_intersample, softmax_mod),
              torch::nn_dropout(p = attn_dropout),
              torch::nn_layer_norm(dim * cols),
              ff(dim * cols, dropout = ff_dropout),
              torch::nn_layer_norm(dim * cols)
            )
          )
        )
      }
    } else {
      for (i in 1:depth){
        self$layers$append(
          torch::nn_module_list(
            list(
              attention(dim, heads_selfattn, dim_heads_selfattn, softmax_mod),
              torch::nn_dropout(attn_dropout),
              torch::nn_layer_norm(dim),
              ff(dim, dropout = ff_dropout),
              torch::nn_layer_norm(dim)
            )
          )
        )
      }
    }

  },
  forward = function(x){
    if (self$intersample){
      for (i in 1:length(self$layers)){
        y <- self$layers[[i]][[1]](x)
        y <- self$layers[[i]][[2]](y)
        x <- self$layers[[i]][[3]](y)$add_(x)
        x <- self$layers[[i]][[4]](x)
        x <- self$layers[[i]][[5]](x)

        # change the shape for intersample attention
        b <- x$shape[1]
        n <- x$shape[2]
        d <- x$shape[3]
        x <- x$reshape(c(1, b, n*d))
        y <- self$layers[[i]][[6]](x)
        y <- self$layers[[i]][[7]](y)
        x <- self$layers[[i]][[8]](y)$add_(x)
        x <- self$layers[[i]][[9]](x)
        x <- self$layers[[i]][[10]](x)
        # revert shape
        x <- x$reshape(c(b, n, d))
      }

    } else {
      for (i in 1:length(self$layers)){
        y <- self$layers[[i]][[1]](x)
        y <- self$layers[[i]][[2]](y)
        x <- self$layers[[i]][[3]](y)$add_(x)
        x <- self$layers[[i]][[4]](x)
        x <- self$layers[[i]][[5]](x)
      }
    }

    x
  },

  get_attention = function(x){
    if (self$intersample){
      for (i in 1:length(self$layers)){
        out <- self$layers[[i]][[1]](x, return_attention = TRUE)

        attention_maps <- out[[2]]
        y <- out[[1]]

        y <- self$layers[[i]][[2]](y)
        x <- self$layers[[i]][[3]](y)$add_(x)
        x <- self$layers[[i]][[4]](x)
        x <- self$layers[[i]][[5]](x)

        # change the shape for intersample attention
        b <- x$shape[1]
        n <- x$shape[2]
        d <- x$shape[3]
        x <- x$reshape(c(1, b, n*d))
        out <- self$layers[[i]][[6]](x, return_attention = TRUE)

        is_attention_maps <- out[[2]]
        y <- out[[1]]

        y <- self$layers[[i]][[7]](y)
        x <- self$layers[[i]][[8]](y)$add_(x)
        x <- self$layers[[i]][[9]](x)
        x <- self$layers[[i]][[10]](x)
        # revert shape
        x <- x$reshape(c(b, n, d))

        list(x, list(attention_maps, is_attention_maps))
      }

    } else {
      for (i in 1:length(self$layers)){
        out <- self$layers[[i]][[1]](x, return_attention = TRUE)

        attention_maps <- out[[2]]
        y <- out[[1]]

        y <- self$layers[[i]][[2]](y)
        x <- self$layers[[i]][[3]](y)$add_(x)
        x <- self$layers[[i]][[4]](x)
        x <- self$layers[[i]][[5]](x)

        list(x, attention_maps)
      }
    }
  }
)

tabular_mlp <- torch::nn_module(
  "tabular_mlp",
  initialize = function(dims, type="binary"){
    dim_pairs <- into_pairs(dims)
    layers <- lapply(dim_pairs, function(x) torch::nn_linear(x[1],x[2]))

    self$mlp <- torch::nn_sequential()
    mapply(function(x, y) self$mlp$add_module(name = x, module = y), names(layers), layers)

  },
  forward = function(x)
    self$mlp(x)
)

#' Tabtransformer
#'
#' @param categories a vector containing the dimensions of each categorical predictor (in the correct order)
#' @param num_continuous the number of continuous predictors
#' @param dim_out dimensions of the output (default is 1)
#' @param intersample boolean value designating whether to use intersample attention
#' @param dim embedding dimension for categorical and continuous data
#' @param depth number of transformer layers
#' @param heads_selfattn number of self-attention heads
#' @param heads_intersample number of intersample attention heads
#' @param dim_heads_selfattn dimensions of the self-attention heads
#' @param dim_heads_intersample dimension of the intersample attention heads
#' @param attn_dropout dropout percentage for attention layers
#' @param ff_dropout dropout percentage for feed-forward layers
#' @param mlp_hidden_mult a numerical vector indicating the hidden dimensions of the final MLP
#' @param softmax_mod multiplier for the attention softmax function
#' @param device 'cpu' or 'cuda'

#'
#' @return a tabtransformer model
#' @export
#'
#' @examples
tabtransformer <- torch::nn_module(
  "tabtransformer",
  initialize = function(
    categories,
    num_continuous,
    dim_out = 1,
    intersample = TRUE,
    dim = 16,
    depth = 4,
    heads_selfattn = 8,
    heads_intersample = 8,
    dim_heads_selfattn = 8,
    dim_heads_intersample = 8,
    attn_dropout = 0.1,
    ff_dropout = 0.8,
    mlp_hidden_mult = c(4, 2),
    softmax_mod = 1,
    device = 'cuda'
  ) {
    self$dim <- dim
    self$dim_out <- dim_out
    self$num_continuous <- num_continuous
    self$depth <- depth
    self$heads_selfattn <- heads_selfattn
    self$heads_intersample <- heads_intersample
    self$dim_heads_selfattn <- dim_heads_selfattn
    self$dim_heads_intersample <- dim_heads_intersample
    self$attn_dropout <- attn_dropout
    self$ff_dropout <- ff_dropout
    self$intersample <- intersample
    self$softmax_mod <- softmax_mod
    self$device <- device

    num_categorical <- length(categories)
    num_unique_categories <- sum(categories)

    total_tokens <- num_unique_categories + 2

    categories_offset <- nnf_pad(torch_tensor(categories, device = self$device), pad = c(1,0), value = 2)
    categories_offset <- categories_offset$cumsum(dim=1)
    lco <- length(categories_offset) - 1
    categories_offset <- categories_offset[1:lco]
    self$register_buffer("categories_offset", categories_offset)

    self$cols <- num_categorical + num_continuous

    # Layers

    self$embeds_cat <- nn_embedding(total_tokens, self$dim)
    self$embeds_cont <- nn_module_list(
      lapply(1:self$num_continuous, function(x) continuous_embedding(100, self$dim))
    )

    self$norm <- nn_layer_norm(num_continuous)
    self$transformer <- tabular_transformer(
      dim = self$dim,
      cols = self$cols,
      depth = self$depth,
      heads_selfattn = self$heads_selfattn,
      heads_intersample = self$heads_intersample,
      dim_heads_selfattn = self$dim_heads_selfattn,
      dim_heads_intersample = self$dim_heads_intersample,
      attn_dropout = self$attn_dropout,
      ff_dropout = self$ff_dropout,
      softmax_mod = self$softmax_mod,
      intersample = self$intersample
    )

    input_size <- self$dim * self$cols
    l <- floor(input_size / 8)

    hidden_dims <- mlp_hidden_mult * l
    all_dims <- c(input_size, hidden_dims, self$dim_out)

    self$mlp <- tabular_mlp(all_dims)

  },
  forward = function(x){
    x_cat <- x$x_cat
    x_cont <- x$x_cont


    ## Insert test of cat size
    x_cat <- x_cat + self$categories_offset
    x_cat <- self$embeds_cat(x_cat)

    ## Insert test of cont size
    x_cont <- self$norm(x_cont)
    n <- x_cont$shape

    x_cont_enc <- torch::torch_empty(n[[1]], n[[2]], self$dim, device = self$device)

    for (i in 1:self$num_continuous) {
      x_cont_enc[,i,] <- self$embeds_cont[[i]](x_cont[,i])
    }

    x <- torch::torch_cat(c(x_cat, x_cont_enc), dim = 2)
    x <- self$transformer(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$mlp(x)
    x
  },
  predict_attn = function(x){
    x_cat <- x$x_cat
    x_cont <- x$x_cont


    ## Insert test of cat size
    x_cat <- x_cat + self$categories_offset
    x_cat <- self$embeds_cat(x_cat)

    ## Insert test of cont size
    x_cont <- self$norm(x_cont)
    n <- x_cont$shape

    x_cont_enc <- torch::torch_empty(n[[1]], n[[2]], self$dim, device = self$device)

    for (i in 1:self$num_continuous) {
      x_cont_enc[,i,] <- self$embeds_cont[[i]](x_cont[,i])
    }

    x <- torch::torch_cat(c(x_cat, x_cont_enc), dim = 2)
    out <- self$transformer$get_attention(x)
    # x <- torch_flatten(out[[1]], start_dim = 2)
    # x <- self$mlp(x)
    list(x, out[[2]])
  }
)
