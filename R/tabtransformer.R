#' Tabtransformer
#'
#' A torch \code{\link[torch]{nn_module}} using multi-headed self attention (MHSA) for tabular datasets.
#' Additionally, an intersample attention (between rows) layer will be added by setting \code{intersample = FALSE}.
#'
#' \href{https://arxiv.org/abs/2012.06678}{Huang et al.} introduce MHSA for tabular datasets,
#' \href{https://arxiv.org/abs/2106.01342}{Somepalli et al.} introduce the concept of intersample attention.
#'
#' @param categories (int vector) a vector containing the dimensions of each categorical predictor (in the correct order)
#' @param num_continuous (int) the number of continuous predictors
#' @param dim_out (int) dimensions of the output (default is 1, matching the default binary task)
#' @param final_layer (nn_module) the final layer of the model (e.g. \code{nn_relu()} to constrain
#' output to values >= 0 only). Default is NULL, which results a in \code{nn_identity()} layer.
#' @param attention (str) string value indicating which type(s) of attention to
#' use, either "both", "mhsa" or "intersample". Default: "both"
#' @param attention_type (str) string value indicating either traditional softmax
#' attention ("softmax") or signed attention ("signed"), which preserves the sign
#' of the attention heads (negative or positive), so that attention heads can
#' be interpreted as either being positively or negatively correlated with the
#' outcome.
#' @param is_first (bool) designates whether intersample attention comes before MHSA
#' @param dim (int) embedding dimension for categorical and continuous data
#' @param depth (int) number of transformer layers
#' @param heads_selfattn (int) number of self-attention heads
#' @param heads_intersample (int) number of intersample attention heads
#' @param dim_heads_selfattn (int) dimensions of the self-attention heads
#' @param dim_heads_intersample (int) dimension of the intersample attention heads
#' @param attn_dropout (float) dropout percentage for attention layers. Default: 0.1.
#' @param ff_dropout (float) dropout percentage for feed-forward layers between attention layers. . Default: 0.1.
#' @param embedding_dropout (float) dropout after the embedding layer. Default: 0.1.
#' @param mlp_dropout (float) dropout between MLP layers. Default: 0.1.
#' @param mlp_hidden_mult (int vector) a numerical vector indicating the hidden dimensions of the final MLP
#' @param softmax_mod (float) multiplier for the MHSA softmax function
#' @param is_softmax_mod (floart) multiplier for the intersample attention softmax function
#' @param device (str) 'cpu' or 'cuda'

#'
#' @return a tabtransformer model
#' @export
#'
#' @examples
#'
#' tabtransformer(
#'   categories = c(4, 2, 13),
#'   num_continuous = 6,
#'   final_layer = nn_relu(),
#'   depth = 1,
#'   dim = 32
#'   )


tabtransformer <- torch::nn_module(
  "tabtransformer",
  initialize = function(
    categories,
    num_continuous,
    dim_out = 1,
    final_layer = NULL,
    attention = "both",
    attention_type = "softmax",
    is_first = FALSE,
    dim = 16,
    depth = 4,
    heads_selfattn = 8,
    heads_intersample = 8,
    dim_heads_selfattn = 8,
    dim_heads_intersample = 8,
    attn_dropout = 0.1,
    ff_dropout = 0.8,
    embedding_dropout = 0.1,
    mlp_dropout = 0.1,
    mlp_hidden_mult = c(4, 2),
    softmax_mod = 1,
    is_softmax_mod = 1,
    device = 'cuda'
  ) {
    if (!(attention %in% c("both", "mhsa", "intersample"))){
      stop("attention must be one of both, mhsa or intersample")
    }
    self$dim <- dim
    self$dim_out <- dim_out
    self$final_layer <- final_layer
    self$num_continuous <- num_continuous
    self$depth <- depth
    self$heads_selfattn <- heads_selfattn
    self$heads_intersample <- heads_intersample
    self$dim_heads_selfattn <- dim_heads_selfattn
    self$dim_heads_intersample <- dim_heads_intersample
    self$attn_dropout <- attn_dropout
    self$ff_dropout <- ff_dropout
    self$embedding_dropout <- embedding_dropout
    self$mlp_dropout <- mlp_dropout
    self$attention <- attention
    self$attention_type <- attention_type
    self$is_first <- is_first
    self$softmax_mod <- softmax_mod
    self$is_softmax_mod <- is_softmax_mod
    self$device <- device


    self$num_categorical <- length(categories)
    num_unique_categories <- sum(categories)

    total_tokens <- num_unique_categories + 2

    if (is.null(categories)) categories <- 0

    categories_offset <- nnf_pad(torch_tensor(categories, device = self$device), pad = c(1,0), value = 2)
    categories_offset <- categories_offset$cumsum(dim=1)
    lco <- length(categories_offset) - 1
    categories_offset <- categories_offset[1:lco]$to(dtype = torch::torch_long())
    self$register_buffer("categories_offset", categories_offset)

    self$cols <- self$num_categorical + num_continuous

    # Layers

    self$embeds_cat <- nn_embedding(total_tokens, self$dim)
    self$embeds_cont <- nn_module_list(
      lapply(1:self$num_continuous, function(x) continuous_embedding(100, self$dim))
    )

    #self$dropout <- nn_dropout(self$embedding_dropout)

    self$norm <- nn_layer_norm(num_continuous)

    if (self$attention == "both"){
      if (self$is_first){
        self$transformer <- tabular_transformer_combined_isfirst(
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
          is_softmax_mod = self$is_softmax_mod,
          attention_type = self$attention_type
        )
      } else {
        self$transformer <- tabular_transformer_combined_islast(
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
          is_softmax_mod = self$is_softmax_mod,
          attention_type = self$attention_type
        )
      }
    } else if (self$attention == "intersample") {
      self$transformer <- tabular_transformer_intersample(
        dim = self$dim,
        cols = self$cols,
        depth = self$depth,
        heads_selfattn = self$heads_selfattn,
        heads_intersample = self$heads_intersample,
        dim_heads_selfattn = self$dim_heads_selfattn,
        dim_heads_intersample = self$dim_heads_intersample,
        attn_dropout = self$attn_dropout,
        ff_dropout = self$ff_dropout,
        is_softmax_mod = self$is_softmax_mod,
        attention_type = self$attention_type
      )
    } else if (self$attention == "mhsa") {
      self$transformer <- tabular_transformer_mhsa(
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
        attention_type = self$attention_type
      )
    } else {
      stop("no appropriate attention type(s) selected")
    }

    input_size <- self$dim * self$cols
    l <- floor(input_size / 8)

    hidden_dims <- mlp_hidden_mult * l
    all_dims <- c(input_size, hidden_dims, self$dim_out)

    self$mlp <- tabular_mlp(all_dims, self$final_layer, self$mlp_dropout)

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

    if (self$num_continuous > 0){
      for (i in 1:self$num_continuous) {
        x_cont_enc[,i,] <- self$embeds_cont[[i]](x_cont[,i])
      }
    }

    x <- torch::torch_cat(c(x_cat, x_cont_enc), dim = 2)
    #x <- self$dropout(x)
    x <- self$transformer(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$mlp(x)
    x
  },
  predict_attn = function(x, intersample = FALSE){
    x_cat <- x$x_cat
    x_cont <- x$x_cont


    ## Insert test of cat size
    x_cat <- x_cat + self$categories_offset
    x_cat <- self$embeds_cat(x_cat)

    ## Insert test of cont size
    # x_cont <- self$norm(x_cont)
    n <- x_cont$shape

    x_cont_enc <- torch::torch_empty(n[[1]], n[[2]], self$dim, device = self$device)

    if (self$num_continuous > 0){
      for (i in 1:self$num_continuous) {
        x_cont_enc[,i,] <- self$embeds_cont[[i]](x_cont[,i])
      }
    }

    x <- torch::torch_cat(c(x_cat, x_cont_enc), dim = 2)
    out <- self$transformer$get_attention(x, intersample = intersample)
    x <- torch_flatten(out[[1]], start_dim = 2)
    x <- self$mlp(x)
    list(x, out[[2]])
  }
)
