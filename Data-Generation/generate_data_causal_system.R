
#' Generate a model
#' 
#' Generate a causal dynamical models with a DAG as underlying causal graph
#' @param p integer, the number of variables in the model
#' @param maxp integer, the maximum number of causal parents 
#' @return a list which include a function that can be used to generate data 
gen_model <- function(p, maxlag = 1, maxp = p, sd = 0.1){
  ## 
  noise <- function(n) rnorm(n, sd = sd)
  ## generate the variable casual order
  order <- sample(p)
  var_names <- paste0("X", order)
  ## for each variable choose randomly some parents among the previous variables
  ## also randomly choose some lags for each parent
  parents <- lapply(1:p, function(i){
    vv <- var_names[i]
    previous <- c()
    if (i>1) previous <- var_names[1:(i - 1)]
    pr <- sample(previous, sample(min(i - 1, maxp), size = 1))
    sapply(pr, function(pp) sample(maxlag, 1), USE.NAMES = TRUE)
  })
  names(parents) <- var_names
  
  parents[[1]] <- c(0)
  
  
  # hasta ací tinc el grafo creat. On cada variable té els seus pares 
  
  #print(parents)
  auto_lags <- sample(maxlag, p, replace = TRUE)
  names(auto_lags) <- var_names
  ## generate regression functions for each variable
  regress <- lapply(1:p, function(i){
    ## generate lag for autoregression
    lag_auto <- 1 + sample(maxlag - 1, 1)
    vv <- var_names[i]
    lags <- parents[[vv]]
    parents <- names(lags)
    d_input <- sum(lags) + lag_auto ## dimension input of the function
    ## now we need to generate a function that take d_input and return a scalar
    Rd <- data.frame(matrix(runif(10*d_input^2 + d_input, min = -1), ncol = d_input))
    colnames(Rd) <- paste0("X", 1:d_input)
    Rd$y <- runif(nrow(Rd), min = -1)
    #print(Rd)
    ### here we fit a linear model with up to 3-way interaction + ploynomial 
    ### of degree 3
    ### you can change this to make it more non-lineal
    ### e.g. you could add exp, sin, cos etc..
    polyterm <- paste0('I(',paste0("X", 1:d_input),"^3)")
    formula <- as.formula(paste0("y ~  .^3 + ", paste(polyterm, collapse = "+")))
    model <- lm(formula = formula, data = Rd)
    ## we need a function that takes the past data and generate a new observation
    function(past){ ## past is an array of past maxlag observations
      input <- c(past[maxlag:(maxlag - lag_auto + 1), vv]) ## past of the variable 
      if (!is.null(parents)){                ## past of the parents
        input <- c(input,  unlist(sapply(parents, function(ppp) {
          lag <- as.numeric(lags[ppp])
          #print(lag)
          past[maxlag:(maxlag - lag + 1), ppp]})))
      }
      input <- data.frame(t(input))
      colnames(input) <- paste0("X", 1:d_input)
      #print(input)
      out <- predict(model, newdata = input)  # podria canviar
      return(min(max(out, -1), 1))  # podria intentar canviar
    }
  })
  
  names(regress) <- var_names
  fun_model <- function(n){
    ## we start from some noise
    DD <- matrix(noise( (maxlag + n) * p), nrow = n + maxlag, ncol = p)
    colnames(DD) <- paste0("X", 1:p)
    for (t in 1:n){
      for (j in 1:p){
        vv <- var_names[j]
        #print(vv)
        DD[maxlag + t, vv] <- 0.3 * DD[maxlag + t - 1, vv] + noise(1) + 
          regress[[vv]](DD[t:(maxlag + t - 1), ,drop = FALSE])
      }
    }
    return(data.frame(DD[-(1:maxlag),]))
  }
  return(list(generate_data = fun_model, parents = parents, 
              var_names = var_names, regress = regress))
}

######################## example #####################
### set seed so I can make example
set.seed(223)

# how many variables:
variables <- c(4,8,12,18,20,30,50)

for(c in variables){
  p <- c
  
  
  maxlags <- "5"
  
  maxps <- "5"
  
  ps <- as.character(p)
  
  datos_nom <-paste0("Datos_maxlag", maxlags, "maxp", maxps, "var",ps,".csv")
  
  matriu_nom <- paste0("Ground_truth_maxlag", maxlags, "maxp", maxps, "var",ps,".txt")
  
  
  model <- gen_model(p, maxlag = 5, maxp = 5, sd = 1)
  
  ## you can generate data with the generate_data function 
  data <- model$generate_data(10000)
  
  ## parents structure is saved here  
  grafo <- model$parents
  
  ###########
  
  # tengo que hacer una función que a partir de los model$parents, construya la matriz de causalidad
  
  vec <- c()
  noms <- paste0("X", 1:p)
  for(i in noms){
    aux <- grafo[grafo = i]
    vec <- c(vec,aux)
  }
  
  
  # matriz de zeros  de p x p
  matriu <- matrix(data =0, nrow = p, ncol = p)
  
  
  
  i <- 0
  j <- 0
  for(element in vec){
    i <- i+1
    for (col in names(element)){
      j <- 0
      for(t in noms ){
        j <- j+1
        if(t == col){matriu[i,j] <- 1}
        
      } 
      
    }
    
  }
  
  matriu <- t(matriu)
  
  ######### matriu  : es ja la matriu de causalitat 
  
  
  ### improtamos datos y la matriz de causalidad : 
  
  write.table(x = matriu, file = matriu_nom, sep = ",", 
              row.names = FALSE, col.names = FALSE)
  
  
  ## to read : 
  
  # matriz_proba<- read.table(file = "matriu.txt", header = FALSE, sep = ",")
  
  write.csv(x = data, file = datos_nom, row.names = FALSE)
  
  
  
}

p <- 10


maxlags <- "5"

maxps <- "5"

ps <- as.character(p)

nom <-paste0("Datos_maxlag", maxlags, "maxp", maxps, "var",ps,".csv")

matriu_nom <- paste0("Ground_truth_maxlag", maxlags, "maxp", maxps, "var",ps,".txt")
model <- gen_model(p, maxlag = 10, maxp = 5, sd = 1)

## you can generate data with the generate_data function 
data <- model$generate_data(1000)

## parents structure is saved here
grafo <- model$parents

###########

# tengo que hacer una función que a partir de los model$parents, construya la matriz de causalidad

vec <- c()
noms <- paste0("X", 1:p)
for(i in noms){
  aux <- grafo[grafo = i]
  vec <- c(vec,aux)
}


# matriz de zeros  de p x p
matriu <- matrix(data =0, nrow = p, ncol = p)



i <- 0
j <- 0
for(element in vec){
  i <- i+1
  for (col in names(element)){
    j <- 0
    for(t in noms ){
      j <- j+1
      if(t == col){matriu[i,j] <- 1}
        
    } 
    
  }

}

matriu <- t(matriu)

######### matriu  : es ja la matriu de causalitat 


### improtamos datos y la matriz de causalidad : 

write.table(x = matriu, file = matriu_nom, sep = ",", 
            row.names = FALSE, col.names = FALSE)


## to read : 

# matriz_proba<- read.table(file = "matriu.txt", header = FALSE, sep = ",")

write.csv(x = data, file = nom, row.names = FALSE)


# to read : 

datos_proba <- read.csv("data.csv")

### try plotting parents and child time series to see if they are really related
plot(scale(data$X0)[200:300], type = "l")
lines(-scale(data$X7)[200:300], col = "red")  ## X7 is parent of X4
 
plot(data$X7[1:499], data$X0[2:500])
