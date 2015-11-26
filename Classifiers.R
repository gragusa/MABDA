library(nnet)
library(ada)
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
glm_1 <- glm.fit(X_scaledTrain_, y_train, family = binomial(link = "logit"))

fitted_glm <- predict_logit(X_scaledTrain_, coef(glm_1))

## This should converge in one iteration, since the glm 
## coefficient set the gradient == 0
coeff <- gradd_logit(y_train, X_scaledTrain_, a = 0.9, 
                     theta0 = coef(glm_1), debug = TRUE, 
                     epsilon = 1e-10, max_iter = 20000)

## This is calculate the regularized version. Try to change lambda to see what 
## is the effect of it. Ideally, you want to do this on the train dataset
coeff_reg <- gradd_logit(y_train, X_scaledTrain_, a = .9, 
                     theta0 = coef(glm_1), debug = TRUE, 
                     epsilon = 1e-10, max_iter = 20000, lambda = 0.001)

## Make FITTED with the regularized model
fitted_reg <- predict_logit(X_scaledTrain_, coeff_reg)

## Calculate the FITTED statistics
confusionMatrix(fitted_glm, y_train)
confusionMatrix(fitted_reg, y_train)


## Calculate PREDICTION
predict_reg <- predict_logit(X_scaledTest_, coeff_reg)
predict_glm <- predict_logit(X_scaledTest_, coef(glm_1))

## Calculate statistics of the PREDICTION
confusionMatrix(predict_glm, y_test)
confusionMatrix(predict_reg, y_test)

################################################################################
## Neural network
################################################################################

## Notice that size is the number of neuron in the hidden layer
## in this case 4. "entropy = TRUE" us
out <- nnet(X_scaledTrain[,-c(1)], y_train, size = 22, entropy = TRUE, maxit = 1000)
plotnet(out)
nn_pred <- ifelse(predict(out, newdata = X_scaledTest[,-c(1)])>0.5, 1, 0)
confusionMatrix(nn_pred, y_test)

## This step create a formula object including all the variables in train
## notice that colnames(train)[-c(1,2)] exclude the intercept (because it will 
## be added back anyway) and `income` which is the dependent variable

## Notice also the backtick around variables. This is necessary becasue `model.matrix` 
## expands the name of the factor as _factor name_ _level_name_, so for instance we
## have "education 11th". In order to make this variable parseable we need to enclose it
## between backtick `education 11th`. 

formula <- formula(paste0("income~",  paste0("`", colnames(X_scaledTrain)[-c(1)], "`", 
                                             collapse = "+")))

## Let's fit a Neural Network model. Notice the "hidden = c(2,2)" is equivalent to fit 
## a neuralnetwork with two neurons in two hidden layers.  

## This may take few minutes --- even more depending on your computer speed. On my machine 
## it takes 11 minutes. NOTE: startweights = array(0, 1200) --> this is initializing the weights 
## to 0. If you don't this you may have serious convergence problems.

nn1 <- neuralnet(formula, data = cbind(income= y_train, X_scaledTrain), threshold = .0003, 
                 hidden = 14, err.fct = "ce", 
                 lifesign = "full", linear.output = FALSE, 
                 startweights = coef(out))

nn2 <- neuralnet(formula, data = cbind(income= y_train, X_scaledTrain), threshold = .03, 
                 hidden = c(14, 6), err.fct = "ce", 
                 lifesign = "full", linear.output = FALSE, 
                 startweights = coef(out))

## plot(nn1)  this will plot the graphic representation of 
## the neural net
plotnet(nn1)

fitted_nn <- neuralnet::compute(nn1, X_scaledTrain[,-1])
fitted_nnet <- ifelse(fitted_nn$net.result>0.5, 1, 0)
confusionMatrix(fitted_nnet, y_train)

predicted_nn <- neuralnet::compute(nn1, X_scaledTest[,-1])
predicted_nnet <- ifelse(predicted_nn$net.result>0.455, 1, 0)
confusionMatrix(predicted_nnet, y_test)

