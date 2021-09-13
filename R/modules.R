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

nn_layernorm_skip <- torch::nn_module(
  "layernorm_skip",
  initialize = function(dim, skip=TRUE){
    self$ln <- torch::nn_layer_norm(dim)
    self$skip <- skip
  },
  forward = function(x1, x2)
    if(self$skip){
      self$ln(x1)$add(x2)
    } else{
      self$ln(x1)
    }
)
