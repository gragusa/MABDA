set.seed(1)
x <- matrix(rnorm(1000*50), 1000, 50)
theta <- array(1, c(50,1))
y <- 1 + x%*%theta + rnorm(1000)
X <- cbind(1, x)

y <- ifelse(y>0, 1, 0)


gradd_logit <- function(y, X, theta0, epsilon = 1e-07, a = .1, 
                        lambda = 0) {
  ## X is n x k matrix
  ## y is n x 1 vector
  ## epsilon: criterion --> when the update is smaller the epsilon you quit
  ## theta0: initial value
  #theta <- array(0, c(3,J))  ## 2 x iterations
  #theta[,1] <- theta0
  Delta <- 1000000
  theta_old <- theta0
  n <- NROW(X)
  while (Delta > epsilon) {
    Xt <- X%*%theta_old
    ht <- c(1/(1+exp(Xt)))
    DJ <- -colMeans(ht*X*c(exp(Xt)*(y-1)+y))
    theta <- theta_old - a*DJ - a*lambda*2*theta_old
    Delta <- max((theta-theta_old)^2)
    cat("Iteration:", iter, "Delta:", Delta, "\n")
    theta_old <- theta
  }
  theta
}

predict_logit <- function(X, theta, cutoff = 0.5) {
  h <- 1/(1+exp(-X%*%theta))
  ifelse(h>0.5, 1, 0)
}

## correctly classified
tt2 <- gradd_logit(y, X, array(1, c(51,1)), a = 2.7, epsilon = 1e-12, lambda = 0.1)
mean(apply(cbind(y, predict_logit(X, tt2)), 1, function(u) u[1]==1 && u[2]==1)) +
mean(apply(cbind(y, predict_logit(X, tt2)), 1, function(u) u[1]==0 && u[2]==0))

tt2 <- gradd_logit(y, X, array(1, c(51,1)), a = 2.7, epsilon = 1e-12, lambda = 0)
mean(apply(cbind(y, predict_logit(X, tt2)), 1, function(u) u[1]==1 && u[2]==1)) +
  mean(apply(cbind(y, predict_logit(X, tt2)), 1, function(u) u[1]==0 && u[2]==0))



gradd <- function(y, X, theta0, epsilon = 1e-07, a = .1) {
  ## X is n x k matrix
  ## y is n x 1 vector
  ## epsilon: criterion --> when the update is smaller the epsilon you quit
  ## theta0: initial value
  #theta <- array(0, c(3,J))  ## 2 x iterations
  #theta[,1] <- theta0
  Delta <- 1000000
  theta_old <- theta0
  n <- NROW(X)
  while (Delta > epsilon) {
    u <- X%*%theta_old-y ## <==== Chenge this for logit
    DJ <- t(X)%*%u/n     ## <==== Chenge this for logit
    theta <- theta_old - a*DJ
    Delta <- max((theta-theta_old)^2)
    cat("Iteration:", iter, "Delta:", Delta, "\n")
    theta_old <- theta
  }
  theta
}

theta0 <- c(.31, 0.3, 0.0)

system.time(gradd(y, x, theta0, 1.7122))
