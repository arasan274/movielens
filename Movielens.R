
#Movielens
#author: Arantxa Sanchis
#date: 21/06/2020

#The MovieLens dataset is available as below:
  
#  . [MovieLens 10M dataset] https://grouplens.org/datasets/movielens/10m/
  
#  . [MovieLens 10M dataset - zip file] http://files.grouplens.org/datasets/movielens/ml-10m.zip

#Note: Due to system constraints in downloading and splitting this very large database, 
#we have been provided with the "edx" training set and "validation" test set by the Harvard 
#teaching staff.

#We have uploaded these datasets to "Google Drive" and made it "publicly available to all".
#We will download them from "Google Drive" into a local folder on our system using the code 
#below.
#We can load the datasets into R as below:
  

################################
# Create edx set, validation set
################################
#Installation of R packages which support the project
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(googledrive)) install.packages("googledrive", repos = "http://cran.us.r-project.org")
if(!require(httpuv)) install.packages("httpuv", repos = "http://cran.us.r-project.org")

#Deauthorize i.e do not request for any login credentials
drive_deauth()
drive_user()

#Download datasets from Google drive as below
#edx dataset
downloaded_file_edx <- drive_download(as_id("15Kd9GctAIx4Yl0yy0huY0EobMqgKEyRq"), overwrite = TRUE)
google_file_edx <- downloaded_file_edx$local_path

#Read dataset into RStudio
edx<-readRDS(google_file_edx)

#validation dataset
downloaded_file_val <- drive_download(as_id("10dUZ4iymvcvq6n36pFDgiTmgMgUxNDFQ"), overwrite = TRUE)
google_file_val <- downloaded_file_val$local_path

#Read dataset into RStudio
validation<-readRDS(google_file_val)

#Check for NAs in the dataset:
  ```{r na_check, echo = TRUE}
na_check <-is.na(edx)
sum(na_check)

#We can see the first six rows ("head" function) of the dataset "edx" as below. 
head(edx) 

#We see the summary function to give a "summary" of the subset as below. 
summary(edx)

#We will also view the class of the objects as below.
str(edx)

#We would need to ascertain the number of users and the number of movies in the edx subset 
#as below:
edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))

#The statistical spread of the ratings is shown below.
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "black", fill = "green") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Statistical Spread of ratings") + theme_gray() 

#Mean_movie_ratings_given_by_users
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black", fill="pink") +
  xlab("Mean rating") +
  ylab("Count of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_gray()

#count_of_ratings_given_per_movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black", fill = "green") +
  scale_x_log10() +
  xlab("Count of user ratings") +
  ylab("Count of movies") +
  ggtitle("Count of ratings given per movie") +
  theme_gray() 

#obscure_movies
edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title)  %>%
  summarize(rating = rating, no_of_ratings = count) %>%
  arrange(desc(rating)) %>%
  slice(1:10) %>%
  knitr::kable()

#Distribution_of_ratings_by_users
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black", fill="pink") +
  scale_x_log10() +
  xlab("Number of ratings") + 
  ylab("Count of users") +
  ggtitle("Distribution of ratings by users") +
  theme_gray() 

#RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#We will split the edx dataset into train and test sets to train our algorithms as below:
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
temp <- edx[test_index,]

#Make sure userId and movieId in test set are also in train set
edx_test <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, edx_test)
edx_train <- rbind(edx_train, removed)

rm(test_index, temp, removed)

## 2.4 Models

### I. Movie Rating Model based on the Average

#The expected rating of the underlying dataset lies between 3 and 4 as seen below.
mu <- mean(edx_train$rating)
mu

#obtain the first naive RMSE (very basic):
naive_rmse <- RMSE(edx_test$rating, mu)
naive_rmse

#Now we proceed to represent the results table with the first RMSE obtained:

#Store the RMSE results of the first model
rmse_results <- data_frame(Method = "Movie Rating Model based on the Average", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

### II.  Movie Rating Model based on the Movie Effects

#obtain a histogram where we see that many movies have negative effects based on the skewness.
movie_averages <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
movie_averages %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"),fill=I("blue"),
                         ylab = "Number of movies", main = "Number of movies with the effect b_i")

#This is because of the penalty term introduced to cater for movie effect.

#We can test our model below and will notice that our prediction has improved considerably 
#using this model.
predicted_ratings <- mu +  edx_test %>%
  left_join(movie_averages, by='movieId') %>%
  pull(b_i)
model_movie_effect_rmse <- RMSE(edx_test$rating, predicted_ratings)

#Store the RMSE results of the second model
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie Rating Model based on the Movie Effects",  
                                     RMSE = model_movie_effect_rmse))
rmse_results %>% knitr::kable()

### III. Movie Rating Model based on the Movie & User Effects

#We can observe the distribution of 'user effects' from the histogram below:
user_averages<- edx_train %>% 
  left_join(movie_averages, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))
user_averages %>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"), fill = I("blue"))


#We compute an approximation of user averages
user_averages <- edx_train %>%
  left_join(movie_averages, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#We can add this effect to our model to construct the predictors.
predicted_ratings <- edx_test %>%
  left_join(movie_averages, by='movieId') %>%
  left_join(user_averages, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

#Store the RMSE results of the third model
model_movie_and_user_effects_rmse <- RMSE(edx_test$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="Movie Rating Model based on the Movie & User Effects",  
                                     RMSE = model_movie_and_user_effects_rmse))
rmse_results %>% knitr::kable()

### IV. Regularized Movie Rating Model based on the Movie & User Effects

#We can also use regularization to estimate the user effect. We will now minimize this equation:

#Try out a range of lambda values
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx_train$rating)
  
  b_i <- edx_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(edx_test$rating, predicted_ratings))
})

#We plot RMSE vs lambdas (??'s) to select the optimal lambda (??).
qplot(lambdas, rmses)  

#For the full model, the optimal lambda which minimizes the RMSE can be obtained as below:
  lambda <- lambdas[which.min(rmses)]
  lambda

#Next, we apply the optimal lambda to make the prediction.
   mu <- mean(edx_train$rating)
  
  b_i <- edx_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  
  regularized_model_movie_and_user_effects_rmse <- RMSE(edx_test$rating, predicted_ratings)
  
#Store the RMSE results of the fourth model
rmse_results <- bind_rows(rmse_results,
                data_frame(Method="Regularized Movie Rating Model based on the Movie & User Effects",                     RMSE = regularized_model_movie_and_user_effects_rmse))
rmse_results %>% knitr::kable()

### V. Matrix Factorization Model

#We will use the R package "recosystem" which provides ways to decompose the rating matrix 
#and estimate the user rating with the help of parallel matrix factorization.

if(!require(recosystem)) 
install.packages("recosystem", repos = "http://cran.us.r-project.org")
set.seed(123)

#Convert the train and test sets into recosystem input format
train_data <-  with(edx_train, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))
test_data  <-  with(edx_test,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))
#Create the model object
r <-  recosystem::Reco()

#Select the best tuning parameters
opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30), 
                                       lrate = c(0.1, 0.2),
                                       costp_l2 = c(0.01, 0.1), 
                                       costq_l2 = c(0.01, 0.1),
                                       nthread  = 4, niter = 10))

#Train the algorithm  
r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))

#Calculate the predicted values  
y_hat_reco <-  r$predict(test_data, out_memory())
head(y_hat_reco, 10)

RMSE_matrix_factorization <- RMSE(edx_test$rating, y_hat_reco)

#Store the RMSE results of the fifth model
rmse_results <- bind_rows(rmse_results,
                data_frame(Method="Matrix Factorization Movie Rating Model",                     
                           RMSE = RMSE_matrix_factorization))
rmse_results %>% knitr::kable()

#The training of our algorithms is complete.

##Validation

#We have trained five algorithms and found the most suitable one to be the 
#Matrix Factorization Model as it gave us the lowest RMSE.

#We will now use the complete "edx" dataset to calculate the RMSE in the "validation" dataset. This will determine the accuracy of the prediction of the movie rating. The project goal is achieved if the RMSE stays below the target. 
set.seed(1234)

#Convert "edx" and "validation" datasets to recosystem input format
edx_reco <-  with(edx, data_memory(user_index = userId, 
                                   item_index = movieId, 
                                   rating = rating))
validation_reco  <-  with(validation, data_memory(user_index = userId, 
                                                  item_index = movieId, 
                                                  rating = rating))

#Create the model object
r <-  recosystem::Reco()

#Tune the parameters
opts <-  r$tune(edx_reco, opts = list(dim = c(10, 20, 30), 
                                     lrate = c(0.1, 0.2),
                                     costp_l2 = c(0.01, 0.1), 
                                     costq_l2 = c(0.01, 0.1),
                                     nthread  = 4, niter = 10))

#Train the model
r$train(edx_reco, opts = c(opts$min, nthread = 4, niter = 20))

#Calculate the prediction
y_hat_final_reco <-  r$predict(validation_reco, out_memory())

RMSE_matrix_factorization_val <- RMSE(validation$rating, y_hat_final_reco)

# Update the result table
rmse_results_final <- data_frame(Method="Matrix Factorization Movie Rating Model",                     
                           RMSE = RMSE_matrix_factorization_val)
rmse_results_final %>% knitr::kable()

#Results

#We therefore found the lowest value of RMSE.The RMSE value of the final 
#model "Matrix Factorization Movie Rating Model" is as below:

rmse_results_final %>% knitr::kable()
