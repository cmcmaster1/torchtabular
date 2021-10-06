
#' Get attention heads
#'
#' @param model the tabtransformer model.
#' @param dataset dataset to pass through the model to generate attention heads.
#' @param n number of rows to use from the dataset. Default is 100.
#' @param device whether to run this on 'cpu' or 'cuda'
#'
#' @return average attention heads as a matrix
#' @export
#'
#' @examples
attention_heads <- function(model, dataset, n = 100, device = 'cpu'){
  temp_model <- model$model$to(device = device)
  temp_model$device <- device
  batch <- dataset$.getitem(1:n)

  batch$x$x_cat <- batch$x$x_cat$to(device = device)
  batch$x$x_cont <- batch$x$x_cont$to(device = device)
  full_output <- temp_model$predict_attn(batch$x, intersample = FALSE)
  rm(temp_model)
  gc(verbose = FALSE, full = TRUE)
  attention_matrix <- torch::torch_cat(full_output[[2]])$mean(c(1,2))$detach()$cpu() %>%
    as.matrix()

  names <- colnames(valid_dl$dataset$predictors)

  rownames(attention_matrix) <- names
  colnames(attention_matrix) <- names
  attention_matrix
}


#' Get intersample attention heads
#'
#' @param model the tabtransformer model.
#' @param dataset dataset to pass through the model to generate intersample
#' attention heads.
#' @param n number of rows to use from the dataset. Default is 100.
#' @param device whether to run this on 'cpu' or 'cuda'
#'
#' @return average intersample attention heads as a matrix
#' @export
#'
#' @examples
intersample_attention_heads <- function(model,
                                        dataset,
                                        n = 100,
                                        device = 'cpu'){


  temp_model <- model$model$to(device = device)
  temp_model$device <- device
  batch <- dataset$.getitem(1:n)

  batch$x$x_cat <- batch$x$x_cat$to(device = device)
  batch$x$x_cont <- batch$x$x_cont$to(device = device)
  full_output <- temp_model$predict_attn(batch$x, intersample = TRUE)
  rm(temp_model)
  gc(verbose = FALSE, full = TRUE)
  attention_matrix <- torch::torch_cat(full_output[[2]])$mean(c(1,2))

  as.matrix(attention_matrix$detach()$cpu())
}

