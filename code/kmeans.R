library(ggplot2)
library(dplyr)

# Generate sample data
set.seed(123)
data <- data.frame(x = c(rnorm(50, 2), rnorm(50, -2), rnorm(50, 5)),
                   y = c(rnorm(50, 2), rnorm(50, -2), rnorm(50, 5)))

# Initialize centroids
set.seed(123)
k <- 3
centroids <- data %>% sample_n(k) 


# empty data

ggplot() +
  geom_point(data = data, aes(x = x, y = y), color="black", alpha=.2, size=4) +
  geom_point(data = centroids %>% mutate(cluster=c(1, 2, 3)), 
             aes(x = x, y = y, color = factor(cluster)), size = 8) +
  ggtitle("Step 1: Initial Assignment") +
  theme_minimal() +
  labs(color = "Cluster")

library(here)
ggsave(here("slides", "week6_figs", "kmeans1.png"))



# Function to assign points to the nearest centroid
assign_points_to_centroids <- function(data, centroids) {
  distances <- as.matrix(dist(rbind(data, centroids)))
  distances <- distances[1:nrow(data), (nrow(data) + 1):(nrow(data) + nrow(centroids))]
  closest <- apply(distances, 1, which.min)
  data$cluster <- closest
  return(data)
}

# Function to update centroids based on current assignments
update_centroids <- function(data, k) {
  centroids <- data %>%
    group_by(cluster) %>%
    summarise(x = mean(x), y = mean(y)) %>%
    mutate(cluster=c(1, 2, 3))
  return(centroids)
}

# Plotting function
plot_kmeans <- function(data, centroids, title) {
  ggplot() +
    geom_point(data = data, aes(x = x, y = y, color = factor(cluster)), size=4) +
    geom_point(data = centroids, aes(x = x, y = y, color = factor(cluster)), size = 8) +
    ggtitle(title) +
    theme_minimal() +
    labs(color = "Cluster")
}

# Initial assignment
data <- assign_points_to_centroids(data, centroids)
plot_kmeans(data, centroids %>% mutate(cluster=c(1, 2, 3)), "Step 1: Initial Assignment")
ggsave(here("slides", "week6_figs", "kmeansfinal.png"))


# Iterate through the k-means steps

i=3
for(i in 1:10) {
  old_centroids <- centroids
  centroids <- update_centroids(data, k)
  data <- assign_points_to_centroids(data, centroids)
  
  # Plot each iteration
  plot_kmeans(data, centroids, paste("Step", i + 1))

  # Check for convergence
  if(identical(old_centroids, centroids)) {
    break
  }
}


ggsave(here("slides", "week6_figs", "kmeansfinal.png"))
