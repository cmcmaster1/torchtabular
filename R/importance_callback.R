#' Importance Callback
#'
#' @param names variable names as they appear in the model input x
#' @param top number of variables to plot (top n). Default is 10.
#' @param start_epoch the epoch to start plotting from. Default is 1.


#'
#' @return two outputs saved to the model context (importance_plot and importance_data)
#' @export
#'
#' @examples
var_importance_callback <- luz::luz_callback(
  name = "importance",
  initialize = function(names,
                        top = 10,
                        start_epoch = 1,
                        retain_data = TRUE,
                        retain_plot = TRUE){
    self$names <- names
    self$top <- top
    self$start_epoch <- start_epoch
    self$retain_dat <- retain_data
    self$retain_plot <- retain_plot
  },
  on_valid_batch_end = function() {
    out <- ctx$model$predict_attn(ctx$batch$x, intersample = FALSE)
    attention_matrix <- torch::torch_cat(out[[2]])$mean(c(1,2,3))$cpu() %>%
      as.matrix()

    rownames(attention_matrix) <- self$var_names
    save_attention <- data.frame(attention_matrix)

    epoch_num <- paste0("epoch", ctx$epoch)

    ctx$log("importance", epoch_num, save_attention)
  },
  on_epoch_end = function() {
    epoch_num <- paste0("epoch", ctx$epoch)
    importances <- ctx$records$importance[[epoch_num]]
    average_importance <- rowMeans(as.data.frame(importances))
    ctx$log("feature_importance", "importance", average_importance)
  },
  on_fit_end = function(){
    imp <- as.data.frame(ctx$records$feature_importance$importance)
    rownames(imp) <- self$names
    colnames(imp) <- paste0("epoch", 1:dim(imp)[2])

    plot_data <- tibble::as_tibble(imp, rownames = "var") %>%
      tidyr::pivot_longer(cols = -var) %>%
      dplyr::mutate(epoch = as.numeric(stringr::str_extract(name, "[0-9]+"))) %>%
      dplyr::group_by(epoch) %>%
      dplyr::mutate(rank = dplyr::row_number(-value))

    if (self$retain_data){
      ctx$importance_data <- plot_data
    }


    top_n <- plot_data %>%
      dplyr::filter(epoch == length(imp), rank <= self$top) %>%
      dplyr::pull(var)

    plot_data <- plot_data %>%
      dplyr::filter(var %in% top_n)

    labels_start <- plot_data %>%
      dplyr::filter(epoch == self$start_epoch) %>%
      dplyr::arrange(rank)

    labels_end <- plot_data %>%
      dplyr::filter(epoch == length(imp)) %>%
      dplyr::arrange(rank)

    plot_data <- plot_data %>%
      filter(epoch >= self$start_epoch)

    self$p <- plot_data %>%
      ggplot(aes(x = epoch, y = rank, group = var)) +
      geom_line(aes(color = var, alpha = 1), size = 2) +
      geom_point(aes(color = var, alpha = 1), size = 4) +
      scale_x_continuous(breaks = self$start_epoch:length(imp), expand = c(.05, .05)) +
      scale_y_reverse(breaks = 1:self$top) +
      geom_text(data = labels_start,
                aes(label = var, x = self$start_epoch - 0.5) , hjust = .85, fontface = "bold", color = "#888888", size = 4) +
      geom_text(data = labels_end,
                aes(label = var, x = epoch + 0.5) , hjust = 0.15, fontface = "bold", color = "#888888", size = 4) +
      theme(legend.position = "none") +
      bump_theme() +
      xlab("Epoch") +
      ylab("Rank") +
      ggtitle("Variable Importance")

    if (self$retain_plot){
      ctx$importance_plot <- p
    }
  }
)








