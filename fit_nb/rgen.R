parms = data.frame(mu = runif(n=10000, min=1, max=10), size = runif(n=10000, min=1, max=100))
data = apply(parms, 1, function(x) {
	mu = x[[1]]
	size = x[[2]]
	rnbinom(n=500, mu=mu, size=size)
})
#data = t(data)

library(data.table)
x = melt(data)[,-1]
x = as.data.table(x)

library(MASS)
nb.fit = glm.nb(Var2 ~ value, data=x)
