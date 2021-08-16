
#' Neptune Callback
#'
#' @param project_name name of the project
#' @param name name of the run
#' @param tags vector of tags
#' @param params vector of hyperparameter names to log
#' @param api_token your Neptune API token

#'
#' @return logs metrics to Neptune
#' @export
#'
#' @examples
neptune_callback <- luz::luz_callback(
  name = "neptune_callback",
  initialize = function(project_name = NULL,
                        name = '',
                        tags = NULL,
                        params = NULL,
                        api_token = 'ANONYMOUS'){
    self$project_name <- project_name
    self$name <- name
    self$tags <- tags
    self$params <- params
    self$api_token <- api_token
  },
  on_fit_begin = function(){

    library(neptune)
    init_neptune(project_name = self$project_name,
                 api_token = self$api_token)

    if (!is.null(self$params)){
      hparams <- ctx$hparams[self$params]
    } else{
      hparams <- ctx$hparams
    }


    # Create the experiment and log hyperparameters
    neptune::create_experiment(name=self$name,
                               params = hparams)

    # Add any tags to distinguish this experiment
    if (!is.null(self$tags)) {
      neptune::append_tag(self$tags)
    }
  },
  on_train_end = function(){
    metrics <- ctx$get_metrics("train", ctx$epoch)
    names(metrics) <- paste0(names(metrics), "_train")

    metrics <- unlist(metrics)

    for (i in 1:length(metrics)){
      neptune::log_metric(metric = names(metrics)[i],
                          value = metrics[i])
    }
  },
  on_valid_end = function(){
    metrics <- ctx$get_metrics("valid", ctx$epoch)
    names(metrics) <- paste0(names(metrics), "_valid")

    metrics <- unlist(metrics)

    for (i in 1:length(metrics)){
      neptune::log_metric(metric = names(metrics)[i],
                          value = metrics[i])
    }

  },
  on_fit_end = function(){
    neptune::stop_experiment()
  }
)
