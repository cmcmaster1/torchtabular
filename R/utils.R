into_pairs <- function(vector){
  ln <- length(vector)
  ln_ <- ln - 1
  in_ <- vector[1:ln_]
  out_ <- vector[2:ln]

  names(in_) <- paste0("layer_", 1:ln_)
  mapply(function(x, y) c(x, y),
         in_,
         out_,
         SIMPLIFY = FALSE,
         USE.NAMES = TRUE)
}

bump_theme <- function() {
  # Colors
  color.background = "white"
  color.text = "#22211d"
  # Begin construction of chart
  ggplot2::theme_bw(base_size=15) +
    # Format background colors
    ggplot2::theme(panel.background = ggplot2::element_rect(fill=color.background, color=color.background)) +
    ggplot2::theme(plot.background  = ggplot2::element_rect(fill=color.background, color=color.background)) +
    ggplot2::theme(panel.border     = ggplot2::element_rect(color=color.background)) +
    ggplot2::theme(strip.background = ggplot2::element_rect(fill=color.background, color=color.background)) +
    # Format the grid
    ggplot2::theme(panel.grid.major.y = ggplot2::element_blank()) +
    ggplot2::theme(panel.grid.minor.y = ggplot2::element_blank()) +
    ggplot2::theme(axis.ticks       = ggplot2::element_blank()) +
    # Format the legend
    ggplot2::theme(legend.position = "none") +
    # Format title and axis labels
    ggplot2::theme(plot.title       = ggplot2::element_text(color=color.text, size=20, face = "bold")) +
    ggplot2::theme(axis.title.x     = ggplot2::element_text(size=14, color="black", face = "bold")) +
    ggplot2::theme(axis.title.y     = ggplot2::element_text(size=14, color="black", face = "bold", vjust=1.25)) +
    ggplot2::theme(axis.text.x      = ggplot2::element_text(size=10, vjust=0.5, hjust=0.5, color = color.text)) +
    ggplot2::theme(axis.text.y      = ggplot2::element_text(size=10, color = color.text)) +
    ggplot2::theme(strip.text       = ggplot2::element_text(face = "bold")) +
    # Plot margins
    ggplot2::theme(plot.margin = ggplot2::unit(c(0.35, 0.2, 0.3, 0.35), "cm"))
}

torch_sparse <- function(tensor){
  indices <- torch::torch_nonzero(tensor)$t()
  values <- torch::tensor[indices[1], indices[2]]
  sparse <- torch::torch_sparse_coo_tensor(indices, values, size = tensor$shape)
  sparse
}



torch_pca <- function(tensor, k, device = 'cpu'){
  n <- tensor$size()[1]
  ones <- torch::torch_ones(n)$view(c(n, 1))

  #h <- (1/n) * torch::torch_mm(ones, ones$t())
  h <- torch::torch_zeros(c(n,n))$view(c(n,n))
  H <- torch::torch_eye(n) - h
  H <- H$to(device = device)

  X_centre <- torch::torch_mm(H$to(dtype = torch::torch_double()), tensor$to(dtype = torch::torch_double()))
  usv <- torch::torch_svd(X_centre)

  components <- usv[[3]][1:k]$t()
  gc()
  components
}
