#' Tabular Model.
#'
#' @param x Either a tidymodels recipe or a data.frame
#' @param model model type to be used - default is a transformer type model.
#' @param ... hyperparameters passed to the model specification
#'
#' @return
#' @export
#'
#' @examples
tab_model<- function(x, model, ...){
  UseMethod("tab_model")
  }

tab_model.recipe <- function(x,
                             model = "tabtransformer",
                             ...){
  if (model == "tabtransformer"){
    predictors <- x$var_info %>%
      dplyr::filter(role == "predictor")

    cat_predictors <- predictors %>%
      dplyr::filter(type == "nominal") %>%
      dplyr::pull(variable)

    cont_predictors <- predictors %>%
      dplyr::filter(type == "numeric") %>%
      dplyr::pull(variable)

    categories <- sapply(x$template[cat_predictors], function(x) length(levels(x)))
    num_continuous <- length(cont_predictors)

    tabtransformer(categories = categories,
                   num_continuous = num_continuous,
                   ...)
  }

}
