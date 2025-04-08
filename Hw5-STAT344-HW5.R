# Q4

# Part (a)

set.seed(777)
num_samples <- 300
sample_size <- 1000

# 300 samples of size 1000 of Exp(1)
samples <- matrix(rexp(num_samples * sample_size, rate = 1), 
                  nrow = sample_size, 
                  ncol = num_samples)

single_sample <- samples[, 1]

plot(density(single_sample), 
     main = "Density Plot of One Exponential Sample", 
     col = "blue", 
     lwd = 2)

hist(single_sample, 
     prob = TRUE, 
     main = "Histogram of One Exponential Sample", 
     col = "lightblue")
lines(density(single_sample), col = "red", lwd = 2)

qqnorm(single_sample, main = "Q-Q Plot for Exponential Sample")
qqline(single_sample, col = "red", lwd = 2)


# Part (b)


sample_means <- apply(samples, 2, mean)

sample_means

plot(density(sample_means), 
     main = "Density Plot of Sample Means", 
     col = "blue", 
     lwd = 2)

hist(sample_means, 
     prob = TRUE, 
     main = "Histogram of Sample Means", 
     col = "lightblue")
lines(density(sample_means), col = "red", lwd = 2)

qqnorm(sample_means, main = "Q-Q Plot for Sample Means")
qqline(sample_means, col = "red", lwd = 2)




