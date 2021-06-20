source("R/attention.R")
source("R/data.R")


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


# fitted <- tabular_transformer %>%
#   setup(
#     loss = nn_bce_with_logits_loss(),
#     optimizer = madgrad::optim_madgrad,
#     metrics = list(
#       luz_metric_binary_auroc()
#     )
#   ) %>%
#   set_hparams(categories = mtcars_train$categories,
#               num_continuous = mtcars_train$num_continuous,
#               dim_out = 1,
#               task = "binary",
#               intersample = FALSE,
#               dim = 16,
#               depth = 4,
#               heads_selfattn = 8,
#               heads_intersample = 8,
#               dim_heads_selfattn = 8,
#               dim_heads_intersample = 8,
#               attn_dropout = 0.1,
#               ff_dropout = 0.8,
#               mlp_hidden_mult = c(4, 2),
#               device = 'cpu') %>%
#   set_opt_hparams(lr = 5e-5) %>%
#   fit(train_dl,
#       epochs = 5,
#       valid_data = valid_dl,
#       accelerator = accelerator(device_placement = TRUE, cpu = TRUE),
#       verbose = TRUE)




