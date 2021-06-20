tabular_dataset <- function(x, ...){
  UseMethod("tabular_dataset")
}

tabular_dataset.data.frame <- torch::dataset(
  "tabular_dataset",
  initialize = function(df, response, cat_vars = NULL, cont_vars = NULL) {
    if (is.null(cat_vars)) {
      cat_vars <- names(which(!sapply(df, is.numeric)))
    }

    if (is.null(cont_vars)) {
      cont_vars <- names(which(sapply(df, is.numeric)))
    }

    self$cat_vars <- df[, which(names(df) %in% cat_vars)]
    self$cont_vars <- df[, which(names(df) %in% cont_vars)]
    self$response <- df[[response]]
  },

  .getitem = function(index) {
    response <- torch::torch_tensor(self$response[index])
    x_cat <- torch::torch_tensor(as.numeric(self$cat_vars[index,]))
    x_cont <- torch::torch_tensor(as.numeric(self$cont_vars[index,]))

    list(x = list(x_cat = x_cat, x_cont = x_cont), y = response)
  },

  .length = function() {
    length(self$response)
  }
)

tabular_dataset.recipe <- torch::dataset(
  "tabular_dataset",
  initialize = function(recipe, data) {
    predictors <- recipe$var_info %>%
      dplyr::filter(role == "predictor")

    cat_predictors <- predictors %>%
      dplyr::filter(type == "nominal") %>%
      dplyr::pull(variable)

    cont_predictors <- predictors %>%
      dplyr::filter(type == "numeric") %>%
      dplyr::pull(variable)

    self$categories <- sapply(recipe$template[cat_predictors], function(x) length(levels(x)))
    self$num_continuous <- length(cont_predictors)

    processed <- hardhat::mold(recipe, data)
    self$predictors <- as.matrix(processed$predictors)
    self$outcomes <- as.numeric(as.matrix(processed$outcomes))

    self$cat_vars <- self$predictors[,cat_predictors]
    self$cont_vars <- self$predictors[,cont_predictors]
    self$response <- self$outcomes
  },

  .getitem = function(index) {
    response <- torch::torch_tensor(self$response[index], dtype = torch_long())
    x_cat <- torch::torch_tensor(self$cat_vars[index,])
    x_cont <- torch::torch_tensor(self$cont_vars[index,])

    list(x = list(x_cat = x_cat, x_cont = x_cont), y = response)
  },

  .length = function() {
    length(self$response)
  }
)
