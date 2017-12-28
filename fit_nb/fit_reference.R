library(data.table)
library(fitdistrplus)

mu = function(r, p) {p * r / (1 - p)}
p = function(r, mu) {mu / r + mu}

data = as.matrix(fread("sample_data.tsv", header = F))
data = t(data)
params = fread("sample_params.tsv")
params[, mu := mu(r, p)]


index = 1
# hist(data[index,], prob = T, breaks = 500)

fit = as.data.table(t(apply(data, 1, function(row) {
    fit = tryCatch(fitdist(row, "nbinom")$estimate, error = function(e) list(size = NA, mu = mean(row)))
    fit
})))
fit$size = unlist(fit$size)
fit$mu = unlist(fit$mu)

print("mean difference between estimated and real 'size' parameter:")
print(mean(abs(fit$size - params$r) / pmax(fit$size, params$r), na.rm = T))
