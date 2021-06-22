#' Tabular Dataset.
#'
#' @param x either a recipe from the {recipes} package or a data.frame
#' @param response response variable name (only used if x is a data.frame)
#' @param cat_vars vector of categorical variable names (only used if x is a data.frame)
#' @param cont_vars vector of continuous variable names (only used if x is a data.frame)
#'
#' @return
#' @export
#'
#' @examples
tabular_dataset <- function(x, response, cat_vars, cont_vars){
  UseMethod("tabular_dataset")
}

#' Tabular Dataset with data.frame input.
#'
#' @param x a data.frame
#' @param response response variable name (only used if x is a data.frame)
#' @param cat_vars vector of categorical variable names (only used if x is a data.frame)
#' @param cont_vars vector of continuous variable names (only used if x is a data.frame)
#'
#' @return
#' @export
#'
#' @examples
tabular_dataset.data.frame <- torch::dataset(
  "tabular_dataset",
  initialize = function(x, response, cat_vars = NULL, cont_vars = NULL) {
    if (is.null(cat_vars)) {
      cat_vars <- names(which(!sapply(x, is.numeric)))
    }

    if (is.null(cont_vars)) {
      cont_vars <- names(which(sapply(x, is.numeric)))
    }

    self$cat_vars <- x[, which(names(x) %in% cat_vars)]
    self$cont_vars <- x[, which(names(x) %in% cont_vars)]
    self$response <- x[[response]]
  },

  .getitem = function(index) {
    response <- torch::torch_tensor(self$response[index])
    x_cat <- torch::torch_tensor(as.numeric(self$cat_vars[index,]), dtype = torch_long())
    x_cont <- torch::torch_tensor(as.numeric(self$cont_vars[index,]))

    list(x = list(x_cat = x_cat, x_cont = x_cont), y = response)
  },

  .length = function() {
    length(self$response)
  }
)

#' Tabular Dataset with recipe input.
#'
#' @param x a recipe
#' @param data a dataset to be prepped by the recipe
#'
#' @return
#' @export
#'
#' @examples
tabular_dataset.recipe <- torch::dataset(
  "tabular_dataset",
  initialize = function(x, data) {
    predictors <- x$var_info %>%
      dplyr::filter(role == "predictor")

    cat_predictors <- predictors %>%
      dplyr::filter(type == "nominal") %>%
      dplyr::pull(variable)

    cont_predictors <- predictors %>%
      dplyr::filter(type == "numeric") %>%
      dplyr::pull(variable)

    self$categories <- sapply(x$template[cat_predictors], function(x) length(levels(x)))
    self$num_continuous <- length(cont_predictors)

    processed <- hardhat::mold(x, data)
    self$predictors <- as.matrix(processed$predictors)
    self$outcomes <- as.numeric(as.matrix(processed$outcomes))

    self$cat_vars <- self$predictors[,cat_predictors]
    self$cont_vars <- self$predictors[,cont_predictors]
    self$response <- self$outcomes
  },

  .getitem = function(index) {
    response <- torch::torch_tensor(self$response[index])
    x_cat <- torch::torch_tensor(self$cat_vars[index,], dtype = torch_long())
    x_cont <- torch::torch_tensor(self$cont_vars[index,])

    list(x = list(x_cat = x_cat, x_cont = x_cont), y = response)
  },

  .length = function() {
    length(self$response)
  }
)
