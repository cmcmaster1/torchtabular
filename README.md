
<!-- README.md is generated from README.Rmd. Please edit that file -->

<img src="assets/austin_mos.png" width="400"/>

# torchtabular

<!-- badges: start -->

[![DOI](https://zenodo.org/badge/378582235.svg)](https://zenodo.org/badge/latestdoi/378582235)

<!-- badges: end -->

A package for training transformer models on tabular datasets, using
SAINT and TabTransformer variant models in R using {torch}.

## Installation

You can install torchtabular from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("cmcmaster1/torchtabular")
```

## Example

``` r
library(torchtabular)
library(tidymodels)
library(tidyverse)
library(torch)
library(luz)
library(madgrad)
```

### Check for GPU and assign device

``` r
device <- ifelse(cuda_is_available(), 'cuda', 'cpu')
```

### Load data

The blastchar dataset is included.

``` r
data('blastchar')
glimpse(blastchar)
#> Rows: 7,043
#> Columns: 21
#> $ customerID       <chr> "7590-VHVEG", "5575-GNVDE", "3668-QPYBK", "7795-CFOCW…
#> $ gender           <chr> "Female", "Male", "Male", "Male", "Female", "Female",…
#> $ SeniorCitizen    <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
#> $ Partner          <chr> "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes…
#> $ Dependents       <chr> "No", "No", "No", "No", "No", "No", "Yes", "No", "No"…
#> $ tenure           <dbl> 1, 34, 2, 45, 2, 8, 22, 10, 28, 62, 13, 16, 58, 49, 2…
#> $ PhoneService     <chr> "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", …
#> $ MultipleLines    <chr> "No phone service", "No", "No", "No phone service", "…
#> $ InternetService  <chr> "DSL", "DSL", "DSL", "DSL", "Fiber optic", "Fiber opt…
#> $ OnlineSecurity   <chr> "No", "Yes", "Yes", "Yes", "No", "No", "No", "Yes", "…
#> $ OnlineBackup     <chr> "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "N…
#> $ DeviceProtection <chr> "No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "Y…
#> $ TechSupport      <chr> "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes…
#> $ StreamingTV      <chr> "No", "No", "No", "No", "No", "Yes", "Yes", "No", "Ye…
#> $ StreamingMovies  <chr> "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes…
#> $ Contract         <chr> "Month-to-month", "One year", "Month-to-month", "One …
#> $ PaperlessBilling <chr> "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "No", …
#> $ PaymentMethod    <chr> "Electronic check", "Mailed check", "Mailed check", "…
#> $ MonthlyCharges   <dbl> 29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.7…
#> $ TotalCharges     <dbl> 29.85, 1889.50, 108.15, 1840.75, 151.65, 820.50, 1949…
#> $ Churn            <chr> "No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Y…
```

### Prepare data

First we will convert the target variable into an integer (0 and 1), and
convert characters to factors so that our tabular dataset will identify
them correctly.

``` r
blastchar <- blastchar %>%
  select(-customerID) %>% 
  mutate(across(c(where(is.character), SeniorCitizen), as_factor),
         Churn = as.numeric(Churn) - 1)

glimpse(blastchar)
#> Rows: 7,043
#> Columns: 20
#> $ gender           <fct> Female, Male, Male, Male, Female, Female, Male, Femal…
#> $ SeniorCitizen    <fct> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
#> $ Partner          <fct> Yes, No, No, No, No, No, No, No, Yes, No, Yes, No, Ye…
#> $ Dependents       <fct> No, No, No, No, No, No, Yes, No, No, Yes, Yes, No, No…
#> $ tenure           <dbl> 1, 34, 2, 45, 2, 8, 22, 10, 28, 62, 13, 16, 58, 49, 2…
#> $ PhoneService     <fct> No, Yes, Yes, No, Yes, Yes, Yes, No, Yes, Yes, Yes, Y…
#> $ MultipleLines    <fct> No phone service, No, No, No phone service, No, Yes, …
#> $ InternetService  <fct> DSL, DSL, DSL, DSL, Fiber optic, Fiber optic, Fiber o…
#> $ OnlineSecurity   <fct> No, Yes, Yes, Yes, No, No, No, Yes, No, Yes, Yes, No …
#> $ OnlineBackup     <fct> Yes, No, Yes, No, No, No, Yes, No, No, Yes, No, No in…
#> $ DeviceProtection <fct> No, Yes, No, Yes, No, Yes, No, No, Yes, No, No, No in…
#> $ TechSupport      <fct> No, No, No, Yes, No, No, No, No, Yes, No, No, No inte…
#> $ StreamingTV      <fct> No, No, No, No, No, Yes, Yes, No, Yes, No, No, No int…
#> $ StreamingMovies  <fct> No, No, No, No, No, Yes, No, No, Yes, No, No, No inte…
#> $ Contract         <fct> Month-to-month, One year, Month-to-month, One year, M…
#> $ PaperlessBilling <fct> Yes, No, Yes, No, Yes, Yes, Yes, No, Yes, No, Yes, No…
#> $ PaymentMethod    <fct> Electronic check, Mailed check, Mailed check, Bank tr…
#> $ MonthlyCharges   <dbl> 29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.7…
#> $ TotalCharges     <dbl> 29.85, 1889.50, 108.15, 1840.75, 151.65, 820.50, 1949…
#> $ Churn            <dbl> 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,…
```

We can now split the data into train and test sets.

``` r
split <- initial_split(blastchar)
train <- training(split)
valid <- testing(split)
```

By creating a recipe, the `tabular_dataset` function will automatically
recognise categorical (must be factors) and continuous predictors.

``` r
recipe <- recipe(blastchar, Churn ~ .) %>%
  step_scale(all_numeric_predictors()) %>%
  step_integer(all_nominal_predictors()) %>% 
  step_impute_linear(all_predictors())
```

We can then pass this recipe to `tabular_dataset` with the relevant
split.

``` r
train_dset <- tabular_dataset(recipe, train)
valid_dset <- tabular_dataset(recipe, valid)
```

Finally, we make a dataloader.

``` r
train_dl <- dataloader(train_dset,
                       batch_size = 512,
                       shuffle = TRUE,
                       num_workers=4)

valid_dl <- dataloader(valid_dset,
                       batch_size = 512,
                       shuffle = FALSE,
                       num_workers=4)
```

# Training

We can now train our model using {luz}

``` r
n_epochs <- 20
```

``` r
model_setup <- tabtransformer %>%
  setup(
    loss = nn_bce_with_logits_loss(),
    optimizer = optim_madgrad,
    metrics = list(
      luz_metric_binary_auroc(from_logits = TRUE),
      luz_metric_binary_accuracy_with_logits()
    )
  ) %>%
  set_hparams(categories = train_dset$categories,
              num_continuous = train_dset$num_continuous,
              dim_out = 1,
              attention = "both",
              attention_type = "softmax",
              is_first = TRUE,
              dim = 32,
              depth = 1,
              heads_selfattn = 32,
              heads_intersample = 32,
              dim_heads_selfattn = 16,
              dim_heads_intersample = 64,
              attn_dropout = 0.1,
              ff_dropout = 0.8,
              embedding_dropout = 0.0,
              mlp_dropout = 0.0,
              mlp_hidden_mult = c(4, 2),
              softmax_mod = 1.0,
              is_softmax_mod = 1.0,
              skip = FALSE,
              device = device)
```

``` r
fitted <- model_setup %>% 
  set_opt_hparams(lr = 1e-4) %>% 
  fit(train_dl,
      epochs = n_epochs,
      valid_data = valid_dl,
      verbose = TRUE)
#> Epoch 1/20
#> Train metrics: Loss: 0.588 - AUC: 0.4969 - Acc: 0.7266
#> Valid metrics: Loss: 0.5965 - AUC: 0.7294 - Acc: 0.7342
#> Epoch 2/20
#> Train metrics: Loss: 0.5626 - AUC: 0.5967 - Acc: 0.7353
#> Valid metrics: Loss: 0.4912 - AUC: 0.7984 - Acc: 0.7575
#> Epoch 3/20
#> Train metrics: Loss: 0.4968 - AUC: 0.7533 - Acc: 0.7397
#> Valid metrics: Loss: 0.4972 - AUC: 0.8037 - Acc: 0.7269
#> Epoch 4/20
#> Train metrics: Loss: 0.465 - AUC: 0.7986 - Acc: 0.7734
#> Valid metrics: Loss: 0.4778 - AUC: 0.809 - Acc: 0.7774
#> Epoch 5/20
#> Train metrics: Loss: 0.4534 - AUC: 0.8074 - Acc: 0.7785
#> Valid metrics: Loss: 0.5443 - AUC: 0.8095 - Acc: 0.7609
#> Epoch 6/20
#> Train metrics: Loss: 0.452 - AUC: 0.8083 - Acc: 0.7772
#> Valid metrics: Loss: 0.4498 - AUC: 0.8199 - Acc: 0.7626
#> Epoch 7/20
#> Train metrics: Loss: 0.4464 - AUC: 0.8155 - Acc: 0.7766
#> Valid metrics: Loss: 0.4479 - AUC: 0.82 - Acc: 0.774
#> Epoch 8/20
#> Train metrics: Loss: 0.4409 - AUC: 0.8181 - Acc: 0.7813
#> Valid metrics: Loss: 0.5036 - AUC: 0.8176 - Acc: 0.766
#> Epoch 9/20
#> Train metrics: Loss: 0.4468 - AUC: 0.8175 - Acc: 0.7836
#> Valid metrics: Loss: 0.4427 - AUC: 0.8232 - Acc: 0.7734
#> Epoch 10/20
#> Train metrics: Loss: 0.4403 - AUC: 0.8211 - Acc: 0.7772
#> Valid metrics: Loss: 0.448 - AUC: 0.8209 - Acc: 0.7751
#> Epoch 11/20
#> Train metrics: Loss: 0.4325 - AUC: 0.8288 - Acc: 0.7916
#> Valid metrics: Loss: 0.4518 - AUC: 0.8234 - Acc: 0.7757
#> Epoch 12/20
#> Train metrics: Loss: 0.439 - AUC: 0.8228 - Acc: 0.7777
#> Valid metrics: Loss: 0.4872 - AUC: 0.8128 - Acc: 0.7785
#> Epoch 13/20
#> Train metrics: Loss: 0.4545 - AUC: 0.8091 - Acc: 0.7758
#> Valid metrics: Loss: 0.4407 - AUC: 0.8235 - Acc: 0.7768
#> Epoch 14/20
#> Train metrics: Loss: 0.4394 - AUC: 0.824 - Acc: 0.7861
#> Valid metrics: Loss: 0.4429 - AUC: 0.8236 - Acc: 0.7763
#> Epoch 15/20
#> Train metrics: Loss: 0.4307 - AUC: 0.8263 - Acc: 0.7859
#> Valid metrics: Loss: 0.4957 - AUC: 0.8213 - Acc: 0.7672
#> Epoch 16/20
#> Train metrics: Loss: 0.4344 - AUC: 0.8272 - Acc: 0.788
#> Valid metrics: Loss: 0.4488 - AUC: 0.8233 - Acc: 0.7814
#> Epoch 17/20
#> Train metrics: Loss: 0.4334 - AUC: 0.8293 - Acc: 0.7906
#> Valid metrics: Loss: 0.4746 - AUC: 0.8246 - Acc: 0.7677
#> Epoch 18/20
#> Train metrics: Loss: 0.4321 - AUC: 0.8319 - Acc: 0.7942
#> Valid metrics: Loss: 0.4498 - AUC: 0.8245 - Acc: 0.778
#> Epoch 19/20
#> Train metrics: Loss: 0.4296 - AUC: 0.8319 - Acc: 0.7927
#> Valid metrics: Loss: 0.4389 - AUC: 0.8275 - Acc: 0.7802
#> Epoch 20/20
#> Train metrics: Loss: 0.4303 - AUC: 0.8307 - Acc: 0.7927
#> Valid metrics: Loss: 0.4826 - AUC: 0.8265 - Acc: 0.7655
```
