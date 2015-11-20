library(nnet)
library(neuralnet)
library(dplyr)
library(NeuralNetTools)
library(caret)  ## this package is used for confusionMatrix which 
## gives information about the predicitive 
## performances of the model
gradd_logit <- function(y, X, theta0, epsilon = 1e-07, a = .1, 
                        lambda = 0, debug = FALSE, max_iter = 4000) {
  ## X is n x k matrix
  ## y is n x 1 vector
  ## epsilon: criterion --> when the update is smaller the epsilon you quit
  ## theta0: initial value
  #theta <- array(0, c(3,J))  ## 2 x iterations
  #theta[,1] <- theta0
  Delta <- 1000000
  theta_old <- theta0
  n <- NROW(X)
  iter <- 1
  while (Delta > epsilon & iter < max_iter) {
    Xt <- X%*%theta_old
    #ht <- c(1/(1+exp(Xt)))
    #DJ <- -colMeans(ht*X*c(exp(Xt)*(y-1)+y))
    ht <- c(1/(1+exp(-Xt)))
    DJ <- colMeans(c(ht-y)*X)
    theta <- theta_old - a*DJ - lambda*2*theta_old
    Delta <- max((theta-theta_old)^2)
    iter <- iter + 1
    if (debug) {
      cat("Iteration:", iter, "Delta:", Delta, "\n")
    }
    theta_old <- theta
  }
  cat("Delta:", Delta)
  theta
}

predict_logit <- function(X, theta, cutoff = 0.5) {
  h <- 1/(1+exp(-X%*%theta))
  ifelse(h>cutoff, 1, 0)
}

dt <- read.csv("~/Google Drive/MABDA shared/train.csv")
dt_ <- read.csv("~/Google Drive/MABDA shared/test.csv")

## Remove first column
dt  <- dt[,-1]
dt_ <- dt_[,-1]

dt  <- dt  %>% mutate(income = ifelse(INCOME_CLASS==" <=50K", 0, 1))
dt_ <- dt_ %>% mutate(income = ifelse(INCOME_CLASS==" <=50K", 0, 1))

n_train <- nrow(dt)
dt_all <- rbind(dt, dt_)

## Create a matrix with factor expanded as dummy variables
## Do this on both the training and test dataset so that we can make a prediction
## directly on the test datset arranged as the train dataset

dtm  <- model.matrix(~income+age+race+sex+education+capital.gain+capital.loss+fnlwgt+workclass, data = dt_all)

dtm_ <- model.matrix(~income+age+I(age^2)+I(age^3)+race+sex+education+capital.gain+capital.loss+fnlwgt+I(fnlwgt^2)+I(fnlwgt^3)+workclass, data = dt_all)

train <- dtm[1:n_train,]
test <- dtm[-c(1:n_train),]

train_ <- dtm_[1:n_train,]
test_  <- dtm_[-c(1:n_train),]


X_train <- train[,-2]
y_train <- train[,2]

X_test <- test[,-2]
y_test <- test[, 2]

X_train_ <- train_[,-2]
y_train_ <- train_[,2]

X_test_ <- test_[,-2]
y_test_ <- test_[, 2]


## Here both binary and continuous variables are scaled (X-m)/s. 
## A better approach would be to center and scale continuous variables and
## encode binary input as {-1, 1} instead of {0, 1}

## procValue
procValues    <- preProcess(X_train[,-1], 
                            method = c("center", "scale"))
X_scaledTrain <-  cbind(1, predict(procValues, X_train[,-1]))
X_scaledTest  <-  cbind(1, predict(procValues, X_test[,-1]))

procValues    <- preProcess(X_train_[,-1], 
                            method = c("center", "scale"))
X_scaledTrain_ <-  cbind(1, predict(procValues, X_train_[,-1]))
X_scaledTest_  <-  cbind(1, predict(procValues, X_test_[,-1]))



## Calculate logistic using GLM --- this has only few regressors --- 
## you should try to increase the number of regressors, in particular 
## adding nonlinear term, 
## e.g. I(age^2), I(age^4)) and adding other continuous variables.

## Cross Validation

nsim <- 50
Acc <- array(0, nsim)
set.seed(1)
for (j in 1:nsim) {
  idx <- sample(1:nrow(X_scaledTrain_), 400, replace = TRUE)
  ## Test is of size 400
  XX_test  <- X_scaledTrain_[idx,]
  yy_test  <- y_train_[idx]
  
  ## Train is of size 25000-400
  XX_train <- X_scaledTrain_[-idx,]
  yy_train <- y_train_[-idx]
  
  glm_1 <- glm.fit(XX_train, yy_train, family = binomial(link = "logit"))
  pred  <- predict_logit(XX_test, coef(glm_1))
  Acc[j] <- confusionMatrix(pred, yy_test)[4]$byClass[8]
  cat("splitting:", j, "\n")
}


nn <- nrow(X_scaledTrain_)
nfold    <- 10
obsinfold <- floor(nn/nfold)
lmbdstar <- Acc_fold <- array(0, nfold)
for (j in 0:(nfold-1)) {
  if (j < (nfold-1))
    idx <- (obsinfold*j+1):(obsinfold*j+obsinfold)
  else
    idx <- (obsinfold*j+1):nn
  ## HERE
  idx_validation <- idx[1:floor(obsinfold/2)]
  idx_test       <- idx[(floor(obsinfold/2)+1):length(idx)]
  
  XX_test  <- X_scaledTrain_[idx_test,]
  yy_test  <- y_train[idx_test]
  
  XX_validation  <- X_scaledTrain_[idx_validation,]
  yy_validation  <- y_train_[idx_validation]
  ## Train is of size 25000-400
  XX_train <- X_scaledTrain_[-idx,]
  yy_train <- y_train_[-idx]
  
  lmbd <- c(.01, .02, .03)
  acctmp <- array(0, length(lmbd))
  count <- 1
  for (lambdas in lmbd)  {
  coeff_reg <- gradd_logit(yy_train, XX_train, a = .9, 
                           theta0 = coef(glm_1), debug = TRUE, 
                           epsilon = 1e-10, max_iter = 20000, 
                           lambda = lambdas)
  pred  <- predict_logit(XX_validation, coeff_reg)
  acctmp[count] <- confusionMatrix(pred, yy_validation)[4]$byClass[8]
  count <- count + 1
  }
  
  lambda_star <- lmbd[which.max(acctmp)]
  
  coeff_reg <- gradd_logit(yy_train, XX_train, a = .9, 
                           theta0 = coef(glm_1), debug = TRUE, 
                           epsilon = 1e-10, max_iter = 20000, 
                           lambda = lambda_star)
  
  pred  <- predict_logit(XX_test, coeff_reg)
  Acc_fold[j+1] <- confusionMatrix(pred, yy_test)[4]$byClass[8]
  lmbdstar[j+1] <- lambda_star
}
  
  