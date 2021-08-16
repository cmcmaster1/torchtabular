# combined_isfirst ----------------
# tabular transformer: both attention types, with intersample preceding mhsa
tabular_transformer_combined_isfirst <- torch::nn_module(
  "tabular_transformer_combined_isfirst",
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
    is_softmax_mod)
  {
    self$layers <- torch::nn_module_list()

    for (i in 1:depth){
      self$layers$append(
        torch::nn_module_list(
          list(
            attention(dim, heads_selfattn, dim_heads_selfattn, softmax_mod),
            torch::nn_dropout(p = attn_dropout),
            torch::nn_layer_norm(dim),
            ff(dim, dropout = ff_dropout),
            torch::nn_layer_norm(dim),
            attention(dim * cols, heads_intersample, dim_heads_intersample, is_softmax_mod),
            torch::nn_dropout(p = attn_dropout),
            torch::nn_layer_norm(dim * cols),
            ff(dim * cols, dropout = ff_dropout),
            torch::nn_layer_norm(dim * cols)
          )
        )
      )
    }

  },
  forward = function(x){
    for (i in 1:length(self$layers)){
      # change the shape for intersample attention
      b <- x$shape[1]
      n <- x$shape[2]
      d <- x$shape[3]

      x <- x$reshape(c(1, b, n*d)) # change shape
      y <- self$layers[[i]][[6]](x) # attention
      y <- self$layers[[i]][[7]](y) # dropout
      x <- self$layers[[i]][[8]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[9]](x) # feed forward
      x <- self$layers[[i]][[10]](y)$add_(x) # layernorm + skip connection


      x <- x$reshape(c(b, n, d)) # revert shape
      y <- self$layers[[i]][[1]](x) # attention
      y <- self$layers[[i]][[2]](y) # dropout
      x <- self$layers[[i]][[3]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[4]](x) # feed forward
      x <- self$layers[[i]][[5]](y)$add_(x) # layernorm + skip connection

    }

    x
  },

  get_attention = function(x, intersample = FALSE){
    attn <- c()
    for (i in 1:length(self$layers)){

      # change the shape for intersample attention
      b <- x$shape[1]
      n <- x$shape[2]
      d <- x$shape[3]
      x <- x$reshape(c(1, b, n*d))
      out <- self$layers[[i]][[6]](x, return_attention = TRUE)

      is_attention_maps <- out[[2]]
      y <- out[[1]]

      y <- self$layers[[i]][[7]](y) # dropout
      x <- self$layers[[i]][[8]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[9]](x) # feed forward
      x <- self$layers[[i]][[10]](y)$add_(x) # layernorm + skip connection
      # revert shape
      x <- x$reshape(c(b, n, d))

      out <- self$layers[[i]][[1]](x, return_attention = TRUE)

      attention_maps <- out[[2]]
      y <- out[[1]]

      y <- self$layers[[i]][[2]](y) # dropout
      x <- self$layers[[i]][[3]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[4]](x) # feed forward
      x <- self$layers[[i]][[5]](y)$add_(x) # layernorm + skip connection

      if (intersample){
        attn <- append(attn, is_attention_maps)
      } else{
        attn <- append(attn, attention_maps)
      }
    }

    list(x, attn)
  }
)




# combined_islast ----------------
# tabular transformer: both attention types, with intersample after mhsa
tabular_transformer_combined_islast <- torch::nn_module(
  "tabular_transformer_combined_islast",
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
    is_softmax_mod)
  {
    self$layers <- torch::nn_module_list()

    for (i in 1:depth){
      self$layers$append(
        torch::nn_module_list(
          list(
            attention(dim, heads_selfattn, dim_heads_selfattn, softmax_mod),
            torch::nn_dropout(p = attn_dropout),
            torch::nn_layer_norm(dim),
            ff(dim, dropout = ff_dropout),
            torch::nn_layer_norm(dim),
            attention(dim * cols, heads_intersample, dim_heads_intersample, is_softmax_mod),
            torch::nn_dropout(p = attn_dropout),
            torch::nn_layer_norm(dim * cols),
            ff(dim * cols, dropout = ff_dropout),
            torch::nn_layer_norm(dim * cols)
          )
        )
      )
    }

  },
  forward = function(x){
    for (i in 1:length(self$layers)){
      y <- self$layers[[i]][[1]](x) # attention
      y <- self$layers[[i]][[2]](y) # dropout
      x <- self$layers[[i]][[3]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[4]](x) # feed forward
      x <- self$layers[[i]][[5]](y)$add_(x) # layernorm + skip connection

      # change the shape for intersample attention
      b <- x$shape[1]
      n <- x$shape[2]
      d <- x$shape[3]

      x <- x$reshape(c(1, b, n*d)) # change shape
      y <- self$layers[[i]][[6]](x) # attention
      y <- self$layers[[i]][[7]](y) # dropout
      x <- self$layers[[i]][[8]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[9]](x) # feed forward
      x <- self$layers[[i]][[10]](y)$add_(x) # layernorm + skip connection
      x <- x$reshape(c(b, n, d)) # revert shape
    }
    x
  },

  get_attention = function(x, intersample = FALSE){
    attn <- c()
    for (i in 1:length(self$layers)){

      out <- self$layers[[i]][[1]](x, return_attention = TRUE)

      attention_maps <- out[[2]]
      y <- out[[1]]

      y <- self$layers[[i]][[2]](y) # dropout
      x <- self$layers[[i]][[3]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[4]](x) # feed forward
      x <- self$layers[[i]][[5]](y)$add_(x) # layernorm + skip connection

      # change the shape for intersample attention
      b <- x$shape[1]
      n <- x$shape[2]
      d <- x$shape[3]
      x <- x$reshape(c(1, b, n*d))
      out <- self$layers[[i]][[6]](x, return_attention = TRUE)

      is_attention_maps <- out[[2]]
      y <- out[[1]]

      y <- self$layers[[i]][[7]](y) # dropout
      x <- self$layers[[i]][[8]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[9]](x) # feed forward
      x <- self$layers[[i]][[10]](y)$add_(x) # layernorm + skip connection
      # revert shape
      x <- x$reshape(c(b, n, d))

      if (intersample){
        attn <- append(attn, is_attention_maps)
      } else{
        attn <- append(attn, attention_maps)
      }
    }

    list(x, attn)
  }
)



# intersample ----------------
# tabular transformer: intersample only
tabular_transformer_intersample <- torch::nn_module(
  "tabular_transformer_intersample",
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
    is_softmax_mod)
  {
    self$layers <- torch::nn_module_list()
    for (i in 1:depth){
      self$layers$append(
        torch::nn_module_list(
          list(
            attention(dim * cols, heads_intersample, dim_heads_intersample, is_softmax_mod),
            torch::nn_dropout(p = attn_dropout),
            torch::nn_layer_norm(dim * cols),
            ff(dim * cols, dropout = ff_dropout),
            torch::nn_layer_norm(dim * cols)
          )
        )
      )
    }
  },
  forward = function(x){
    for (i in 1:length(self$layers)){
      # change the shape for intersample attention
      b <- x$shape[1]
      n <- x$shape[2]
      d <- x$shape[3]

      x <- x$reshape(c(1, b, n*d)) # change shape
      y <- self$layers[[i]][[1]](x) # attention
      y <- self$layers[[i]][[2]](y) # dropout
      x <- self$layers[[i]][[3]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[4]](x) # feed forward
      x <- self$layers[[i]][[5]](y)$add_(x) # layernorm + skip connection
      x <- x$reshape(c(b, n, d)) # revert shape
    }
    x
  },

  get_attention = function(x, intersample = FALSE){
    attn <- c()
    for (i in 1:length(self$layers)){
      # change the shape for intersample attention
      b <- x$shape[1]
      n <- x$shape[2]
      d <- x$shape[3]
      x <- x$reshape(c(1, b, n*d))
      out <- self$layers[[i]][[1]](x, return_attention = TRUE)

      attention_maps <- out[[2]]
      y <- out[[1]]

      y <- self$layers[[i]][[2]](y) # dropout
      x <- self$layers[[i]][[3]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[4]](x) # feed forward
      x <- self$layers[[i]][[5]](y)$add_(x) # layernorm + skip connection
      x <- x$reshape(c(b, n, d))

      attn <- append(attn, attention_maps)
    }

    list(x, attn)
  }
)




# mhsa ----------------
# tabular transformer: mhsa only
tabular_transformer_mhsa <- torch::nn_module(
  "tabular_transformer_mhsa",
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
    softmax_mod)
  {
    self$layers <- torch::nn_module_list()
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
  },
  forward = function(x){
    for (i in 1:length(self$layers)){
      y <- self$layers[[i]][[1]](x) # attention
      y <- self$layers[[i]][[2]](y) # dropout
      x <- self$layers[[i]][[3]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[4]](x) # feed forward
      x <- self$layers[[i]][[5]](y)$add_(x) # layernorm + skip connection
    }
    x
  },

  get_attention = function(x, intersample = FALSE){
    attn <- c()
    for (i in 1:length(self$layers)){
      out <- self$layers[[i]][[1]](x, return_attention = TRUE)

      attention_maps <- out[[2]]
      y <- out[[1]]

      y <- self$layers[[i]][[2]](y) # dropout
      x <- self$layers[[i]][[3]](y)$add_(x) # layernorm + skip connection
      y <- self$layers[[i]][[4]](x) # feed forward
      x <- self$layers[[i]][[5]](y)$add_(x) # layernorm + skip connection

      attn <- append(attn, attention_maps)
    }

    list(x, attn)
  }
)
