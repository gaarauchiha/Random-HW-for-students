# Part A
# Q1
Q1 <- ((7.1^2 * 2.7) / sqrt(322.5))^(1/9)
Q1


# Q2
x <- c(12, -3, 0, 14, -7, 4)
y <- c(1, 0, 0.25, 7, 15, 0.6)
Q2<- x^y
Q2


# Q3
u <- function(x, y, z) {
  (x^2 + y^2 + z^2)^(1/4)
}
u(2.2, 3.7, 5.8)

# Q4
A <- c(6, 7, 0, 9, 15)
B <- c("a", "b", "c", "d", "e")
C <- c(-4, -1, -2, 0, -15)
df <- data.frame(A, B, C)
df