attention_heads <- function(model, dataset, n){
  device <- model$model$device
  batch <- dataset$.getitem(1:n)

  batch$x$x_cat <- batch$x$x_cat$to(device = device)
  batch$x$x_cont <- batch$x$x_cont$to(device = device)
  full_output <- model$model$predict_attn(batch$x)

  attention_matrix <- full_output[[2]][[1]]$mean(c(1,2))$cpu() %>%
    as.matrix()

  names <- colnames(valid_dl$dataset$predictors)

  rownames(attention_matrix) <- names
  colnames(attention_matrix) <- names

  attention_matrix
}


intersample_attention_heads <- function(model, dataset, n){
  device <- model$model$device
  batch <- dataset$.getitem(1:n)

  batch$x$x_cat <- batch$x$x_cat$to(device = device)
  batch$x$x_cont <- batch$x$x_cont$to(device = device)
  full_output <- model$model$predict_attn(batch$x)

  attention_matrix <- full_output[[2]][[2]]$mean(c(1,2))$cpu() %>%
    as.matrix()


  attention_matrix
}
