mlflow_callback <- luz::luz_callback(
  name = "mlflow_callback",
  initialize = function(){

  },
  on_fit_begin = function(){
    mlflow::mlflow_start_run()
  },
  on_train_end = function(){
    metrics <- ctx$get_metrics("train", ctx$epoch)
    names(metrics) <- paste0(names(metrics), "_train")

    metrics <- unlist(metrics)

    for (i in 1:length(metrics)){
      mlflow::mlflow_log_metric(key = names(metrics)[i],
                                value = metrics[i],
                                step = ctx$epoch)
    }
  },
  on_valid_end = function(){
    metrics <- ctx$get_metrics("valid", ctx$epoch)
    names(metrics) <- paste0(names(metrics), "_valid")

    metrics <- unlist(metrics)

    for (i in 1:length(metrics)){
      mlflow::mlflow_log_metric(key = names(metrics)[i],
                                value = metrics[i],
                                step = ctx$epoch)
    }

  },
  on_fit_end = function(){
    mlflow::mlflow_end_run()
  }
)
