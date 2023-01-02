
nvar <- 10
nsample <- 400
errsd <- 1

### generate noise, could change that to different noise functions
x <- matrix(0, ncol = nvar, nrow = nsample)

### generate random matrix of coefficients for lag = 1
p1 <- 0.3 ##prob of edge at lag 1
alpha1 <- matrix(runif(nvar^2, min = -1)  * sample(c(0, 1),prob = c(1-p1, p1), 
                                         size = nvar^2, replace = TRUE), nrow = nvar)

alpha1 <- alpha1 / (1 + max(abs(eigen(alpha1)$values)))

### generate random matrix of coefficients for lag = 2
p2 <- 0.2 ##prob of edge at lag 2
alpha2 <- matrix(runif(nvar^2, min = -1)  * sample(c(0, 1), prob = c(1-p2, p2), 
                                         size = nvar^2, replace = T), nrow = nvar)

alpha2 <- alpha2 / (1 + max(abs(eigen(alpha2)$values)))

b0 <- runif(nvar)



"""
Podria generar más alphas y hacer el probelma con más lags, para que fuera algo más complicado. Y ya con esto calcular todas las matrices de causaliad pasarlo a python 
"""


#  %*%  -> matrix multiplication : 

#  cada fila (3 : 400) de la matriz  es la suma del elemento anterioor por alpha1

# + elemento dos  anteirorior por alpha 2 

# + varianza + bias

# en cada iteración : 
for (i in 3:nsample){
  x[i,] <- alpha1 %*% x[i-1,] +  alpha2 %*% x[i-2,] + rnorm(nvar, sd = errsd) + b0
}


plot(x[,1])
### plot lagged variables against each other
plot(x[-(1:1),4], x[1:(nsample-1),1]) ##lag 1  X4 vs X1

plot(x[-(1:2),4], x[1:(nsample-2),1]) ##lag 2  X4 vs X1  


### check the coefficients that are significant here

# estamos haciendo lm de la serie temporal desde 3:400 con la serie temporal con retraso 2 y 1
summary(lm(x[-(1:2),1] ~ x[1:(nsample-2),] + x[2:(nsample-1),]))

### and they should correspond to the non-zeros entries of 


matriz <- matrix(data =1, nrow = nvar, ncol = 1)


alpha2
alpha1

matiz <-  t(abs(alpha2) + abs(alpha1))

matriz <- 



write.csv(x, "linear_VAR_data.csv",row.names = FALSE)
