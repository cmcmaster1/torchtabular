source("R/attention.R")
source("R/data.R")

devtools::load_all()
library(tidymodels)
library(tidyverse)
library(torch)
library(luz)

data("mtcars")
factor_vars <- c("cyl", "am", "gear", "carb")
mtcars <- mtcars %>%
  mutate(across(all_of(factor_vars), as_factor))

split <- initial_split(mtcars)
train <- training(split)
valid <- testing(split)

recipe <- recipe(mtcars, vs ~ .) %>%
  step_scale(all_numeric_predictors()) %>%
  step_integer(all_of(factor_vars))

mtcars_train <- tabular_dataset(recipe, train)
mtcars_valid <- tabular_dataset(recipe, valid)

train_dl <- dataloader(mtcars_train,
                       batch_size = 32,
                       shuffle = TRUE)

valid_dl <- dataloader(mtcars_valid,
                       batch_size = 32,
                       shuffle = TRUE)

model <- tabtransformer(categories = mtcars_train$categories,
                             num_continuous = mtcars_train$num_continuous,
                             dim_out = 1,
                             task = "binary",
                             intersample = FALSE,
                             dim = 16,
                             depth = 4,
                             heads_selfattn = 8,
                             heads_intersample = 8,
                             dim_heads_selfattn = 8,
                             dim_heads_intersample = 8,
                             attn_dropout = 0.1,
                             ff_dropout = 0.8,
                             mlp_hidden_mult = c(4, 2),
                             device = 'cpu')

x <- mtcars_train$.getitem(1:5)$x
model$forward
model$forward(x)

fitted <- tabtransformer %>%
  setup(
    loss = nn_bce_with_logits_loss(),
    optimizer = torch::optim_adam,
    metrics = list(
      luz_metric_binary_auroc()
    )
  ) %>%
  set_hparams(categories = mtcars_train$categories,
              num_continuous = mtcars_train$num_continuous,
              dim_out = 1,
              task = "binary",
              intersample = TRUE,
              dim = 16,
              depth = 1,
              heads_selfattn = 8,
              heads_intersample = 8,
              dim_heads_selfattn = 8,
              dim_heads_intersample = 8,
              attn_dropout = 0.1,
              ff_dropout = 0.1,
              mlp_hidden_mult = c(4, 2),
              device = 'cuda') %>%
  set_opt_hparams(lr = 5e-3) %>%
  fit(train_dl,
      epochs = 10,
      valid_data = valid_dl,
      accelerator = accelerator(device_placement = TRUE, cpu = TRUE),
      verbose = TRUE)

x <- mtcars_train$.getitem(1:5)$x
x_cont <- x$x_cont
x_cat <- x$x_cat

categories_offset <- nnf_pad(torch_tensor(mtcars_train$categories), pad = c(1,0), value = 2)
categories_offset <- categories_offset$cumsum(dim=1)
lco <- length(categories_offset) - 1
categories_offset <- categories_offset[1:lco]
nn_buffer(categories_offset, persistent = TRUE)

num_categorical <- length(mtcars_train$categories)
num_unique_categories <- sum(mtcars_train$categories)

total_tokens <- num_unique_categories + 2

self$cols <- num_categorical + num_continuous

# Layers

embeds_cat <- nn_embedding(total_tokens, 16)
embeds_cont <- nn_module_list(
  lapply(1:6, function(x) continuous_embedding(100, 16))
)

x_cat <- x_cat + categories_offset
x_cat <- embeds_cat(x_cat)

norm <- nn_layer_norm(6)

x_cont <- norm(x_cont)
n <- x_cont$shape

x_cont_enc <- torch::torch_empty(n[[1]], n[[2]], 16)

for (i in 1:6) {
  x_cont_enc[,i,] <- embeds_cont[[i]](x_cont[,i])
}

x <- torch::torch_cat(c(x_cat, x_cont_enc), dim = 2)
