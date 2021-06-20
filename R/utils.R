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
