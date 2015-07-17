mu <- 2
n <- 200
set.seed(56)
## Fake data
smpl <- rchisq(n, df = mu)

## CONFIDENCE INTERVAL (based on asymptotics)
xbar <- mean(smpl)
s <- sd(smpl)v
c(xbar-1.96*s/sqrt(n), xbar+1.96*s/sqrt(n))

## CONFIDENCE INTERVAL (based on bootstrap)
sim <- 10000
out <- array(0, sim)
for(j in 1:sim) {
  idx <- sample(1:200, 200, replace = TRUE)
  out[j] <- mean(smpl[idx])
}
quantile(out, p=c(0.025, 0.975))


## Fake data 2
set.seed(19)
x <- rnorm(100)
y <- 0.3*x + rnorm(100)
plot(x,y)

for (j in 1:sim) {
  idx <- sample(1:100, 100, replace = TRUE)
  out[j] <- cor(y[idx], x[idx])
}
}


sim <- 100000

out <- array(0, sim)
for (i in 1:sim) {

xbar <- mean(smpl)
out[i] <- sqrt(n)*(xbar-mu)/sqrt(4)
}