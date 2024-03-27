##############################################################################
# File-Name: week7_word_embeddings.r
# Date: March 22, 2024
# author: Tiago Ventura
# course: PPOL 6801 - text as data
# topics: unsupervised learning
# Machine: MacOS High Sierra
##############################################################################

# 0 - Introduction ----------------------------------------------------------

## In this code, we will learn how to work with word embeddings in R. 
## This tutorial is inspired by these materials: 

# Chris Bail: https://cbail.github.io/textasdata/word2vec/rmarkdown/word2vec.html
# Emil Hvitfeldt and Julia Silge: https://smltar.com/
# Chris Barrie: https://cjbarrie.github.io/CTA-ED/exercise-6-unsupervised-learning-word-embedding.html
# Pablo Barbera: http://pablobarbera.com/POIR613/code.html


## This code will focus on: 

# -  Generate word embeddings using a simple combination of co-occurence matrix and matrix factorization
# -  Train a local word embedding model via Glove (also co-occurence matrix) and Word2Vec Algorithms (Neural Nets)
# -  Load pre-trained embeddings
# -  Visualize and inspect results
# -  Use embeddings in downstream supervised learning tasks


# Setup

library(tidyverse) # loads dplyr, ggplot2, and others
library(stringr) # to handle text elements
library(tidytext) # includes set of functions useful for manipulating text
library(text2vec) # for word embedding implementation
library(widyr) # for reshaping the text data
library(irlba) # for svd
library(here)
library(quanteda)

# Data
## As in Emil Hvitfeldt and Julia Silge, we will use data from 
## United States Consumer Financial Protection Bureau (CFPB) about Complains on financial products and services. 

## You can download the data here: https://github.com/EmilHvitfeldt/smltar/blob/master/data/complaints.csv.gz

cpts <- read_csv(here("data", "complaints.csv"))

# create an id
cpts$id <- 1:nrow(cpts)

# 1 - Generate Embeddings via Matrix Factorization -----------------------------------------

## this is not the estimation technique we saw in class. But it is a efficient way, similar to the Glove algorithm, for you to
## estimate embeddings locally. More importanly, by estimating your embeddings step by step, I think you will get a better sense 
## of how word vectors work. 

## Here are the steps: 

## Step 1: get the unigram probability for each word: How often do I see word1 and word2 independently?

## Step 2: Skipgram Probability. How often did I see word1 nearby word2? 

## Step 3: Calculate the Normalized Skipgram Probability (or PMI, as we saw before). log (p(word1, word2) / p(word1) / p(word2))

## Step 4: Convert this to a huge matrix with PMI in the cells

## Step 5: Use Singular Value Decomposition to find the word vectors (the rows on the left singular vectors matrix)


## 1.1 - Estimation ----------------
cpts <- cpts %>% slice(1:10000)

# Step 1: get the unigram probabilities

#calculate unigram probabilities (used to normalize skipgram probabilities later)

unigram_probs <- cpts %>%
  select(id,consumer_complaint_narrative) %>% 
  unnest_tokens(word, consumer_complaint_narrative) %>%
  count(word, sort = TRUE) %>%
  # calculate probabilities
  mutate(p = n / sum(n))

## Step 2: Skipgram Probabilities

#create context window with length 6
tidy_skipgrams <- cpts %>% 
  select(id, consumer_complaint_narrative) %>%
  # unnesting the ngrams
  unnest_tokens(ngram, consumer_complaint_narrative, token = "ngrams", n = 6) %>%
  # creating an id for ngram
  mutate(ngramID = row_number()) %>% 
  # create a new id which is pasting id for the comment and id for the ngram
  tidyr::unite(skipgramID, id, ngramID) %>%
  # unnesting again
  unnest_tokens(word, ngram)

# let's see how it looks like
head(tidy_skipgrams, n=20)

## What we need to do now is to calculate the joint probability of word 1 and word 2 across all the windows. 
## basically for every window

tidy_skipgrams <- tidy_skipgrams %>%
  pairwise_count(word, skipgramID, diag = TRUE, sort = TRUE) %>% # diag = T means that we also count when the word appears twice within the window
  mutate(p = n / sum(n))

## Step 3: Get the PMI
head(tidy_skipgrams)
## Join the skipgram with the unigram probabilities
normalized_prob <- tidy_skipgrams %>%
  filter(n > 20) %>%
  rename(word1 = item1, word2 = item2) %>%
  left_join(unigram_probs %>%
              select(word1 = word, p1 = p),
            by = "word1") %>%
  left_join(unigram_probs %>%
              select(word2 = word, p2 = p),
            by = "word2") %>%
  mutate(p_together = p / p1 / p2)

## log the final probability
pmi_matrix <- normalized_prob %>%
  mutate(pmi = log10(p_together)) 


## Step 4 - Convert to a huge matrix
?cast_sparse
pmi_matrix <- pmi_matrix %>%
               cast_sparse(word1, word2, pmi)

#remove missing data
# notice this is a non-standard list in R. It is called S4 object type, and you access the elements using @, instead of $
sum(is.na(pmi_matrix@x))
pmi_matrix@x[is.na(pmi_matrix@x)] <- 0

## Step 5 - Matrix Factorization

# run SVD
pmi_svd <- irlba(pmi_matrix, 256, maxit = 500)

str(pmi_svd)
# Here are your word vectors
word_vectors <- pmi_svd$u
rownames(word_vectors) <- rownames(pmi_matrix)

# let's look at them briefly
word_vectors["error",]


## 1.2 - Analyzing Word Embeddings ----------------

# Let's write a function to get the neared neighbors
nearest_words <- function(word_vectors, word, n){
  selected_vector = word_vectors[word,]
  
  mult = as.data.frame(word_vectors %*% selected_vector) # dot product in R
  mult %>%
    rownames_to_column() %>%
    rename(word = rowname,
           similarity = V1) %>%
    anti_join(get_stopwords(language = "en")) %>%
    arrange(-similarity) %>%
    slice(1: n)
  
}

rownames(word_vectors)
# See some words
nearest_words(word_vectors, "error", 10) 
nearest_words(word_vectors, "month", 10) 
nearest_words(word_vectors, "fee", 10) 

# Visualize these vectors
nearest_words(word_vectors, "error", 15) %>%
  mutate(token = reorder(word, similarity)) %>%
  ggplot(aes(token, similarity)) +
  geom_col(show.legend = FALSE, fill="#336B87")  +
  coord_flip() +
  theme_minimal()

# Since we have found word embeddings via singular value decomposition,
# we can use these vectors to understand what principal components explain the most variation
#in the CFPB complaints. 

# convert to a dataframe
wv_tidy <- word_vectors %>%
  as_tibble() %>%
  mutate(word=rownames(word_vectors)) %>%
  pivot_longer(cols = contains("V"), 
               names_to = "dimension", 
               values_to = "value") %>%
  mutate(dimension=str_remove(dimension, "V"), 
         dimension=as.numeric(dimension))
  

wv_tidy %>%
  # 12 largest dimensions
  filter(dimension <= 12) %>%
  # remove stop and functional words
  anti_join(get_stopwords(), by = "word") %>%
  filter(word!="xxxx", word!="xx") %>%
  # group by dimension
  group_by(dimension) %>%
  top_n(12, abs(value)) %>%
  ungroup()  %>%
  mutate(item1 = reorder_within(word, value, dimension)) %>%
  ggplot(aes(item1, value, fill = dimension)) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~dimension, scales = "free_y", ncol = 4) +
  scale_x_reordered() +
  coord_flip() +
  labs(
    x = NULL,
    y = "Value",
    title = "First 24 principal components for text of CFPB complaints",
    subtitle = paste("Top words contributing to the components that explain",
                     "the most variation")
  )



## visualize two main dimensions for a certain groups of words

#grab 100 words
forplot<-as.data.frame(word_vectors[200:300,])
forplot$word<-rownames(forplot)

#now plot
library(ggplot2)
ggplot(forplot, aes(x=V1, y=V2, label=word))+
  geom_text(aes(label=word),hjust=0, vjust=0, color="#336B87")+
  theme_minimal()+
  xlab("First Dimension Created by SVD")+
  ylab("Second Dimension Created by SVD")


# 2 - Train your word embeddings: GloVe and Word2Vec --------------------

## Instead of using Neural Networks, Glove uses also a co-occurence matrix to build word vectors. 

## These are the main difference between GloVe and Word2vec: 

# - Glove looks at global co-occurence and optimize the parameters based on that. 
# - Word2Vec uses a local optimization, doing self-supervision on local windows. 

# For this reason, Word2Vec requires a lot of data. 
# While GloVe can recover meaningful parameters with less data

## See some sources from these models here: 

# - `Word2Vec` from Mikolov et al, 2013 ([paper](https://arxiv.org/pdf/1310.4546.pdf), [paper](https://arxiv.org/pdf/1301.3781.pdf), [code](https://code.google.com/archive/p/word2vec/)) 
# - `GloVe` from Pennington et al, 2014 ([website](https://nlp.stanford.edu/projects/glove/))

# you can also read here more about these models: 
## - https://github.com/ArthurSpirling/EmbeddingsPaper/blob/master/Project_FAQ/faq.md
## - https://medium.com/cmotions/nlp-with-r-part-2-training-word-embedding-models-and-visualize-results-ae444043e234

# 2.1 - GloVe Training -------------------

## original paper: https://nlp.stanford.edu/pubs/glove.pdf

library(text2vec) # for implementation of GloVe algorithm
library(stringr) # to handle text strings
library(umap) # for dimensionality reduction later on
library(quanteda)

## data: let's work with a sample of congressional records from Spirling and Rodriguez paper
cr = read_rds("data/congresional_records.rds")
dim(cr) # really big. Let's reduce
table(cr$session_id)

# let's get only one session
cr_111 <- cr %>% filter(session_id==111) #%>% sample_n(10000)


## start with some pre-processing
head(cr_111)

# tokenizer as usual
toks <- tokens(cr_111$speech, 
               remove_punct=T, 
               remove_symbols=T, 
               remove_numbers=T, 
               remove_separators=T) 

# only use features that appear at least X times in the corpus
feats <- dfm(toks, tolower=T, verbose = FALSE) %>% 
  dfm_trim(min_termfreq = 10) %>% 
  featnames()

# leave the pads so that non-adjacent words will not become adjacent
toks_feats <- tokens_select(toks,
                            feats,
                            padding = TRUE)
head(toks_feats)


## Defining parameters for the Glove Model

# See more about text2vec here: https://text2vec.org/glove.html 
WINDOW_SIZE <- 6 # size of the windown for counting co-occurence
DIM <- 300 # dimensions of the embeddings
ITERS <- 100 # iterations of the models
COUNT_MIN <- 10 # minimum count of words that we want to keep

# construct the feature co-occurrence matrix for our tokens object
?fcm
toks_fcm <- fcm(toks_feats, 
                context = "window", 
                window = WINDOW_SIZE, 
                count = "frequency", 
                tri = FALSE) # important to set tri = FALSE

head(toks_fcm)

# estimate glove model using text2vec
## set parameters
glove <- GlobalVectors$new(rank = DIM, 
                           x_max = 10,
                           learning_rate = 0.05)

# fit in a pythonic style!
start = Sys.time()

wv_main <- glove$fit_transform(toks_fcm, 
                               n_iter = ITERS,
                               convergence_tol = 1e-3, 
                               n_threads = parallel::detectCores()) # set to 'parallel::detectCores()' to use all available cores
end = Sys.time()
print(end-start)
#saveRDS(wv_main, file = "data/local_glove.rds")
#saveRDS(glove, file = "data/local_glove_context.rds")

wv_main = read_rds(file = "data/local_glove.rds")

# get output
# Note that model learns two sets of word vectors - main and context. \
word_vectors_context <- glove$components
str(glove)

# While both of word-vectors matrices can be used as result it usually better 
# (idea from GloVe paper) to average or take a sum of main and context vector:
word_vectors <- wv_main + t(word_vectors_context) # word vectors

# features?
head(rownames(word_vectors))

class(word_vectors)

# 2.2 -  Pretrained GLoVE embeddings -------------------------------------------

# Notice word meanings should not change much across context. So one of the key ideas of the word embeddings
# is that we can use word embeddings trained on billions of task as a source for our specific tasks

# in Rodriguez and Spirling paper, the authors run a bunch of experiments to show pre-trained embeddings 
# work well compared to locally trained embeddings in political texts. 

# Download Glove here: GloVe pretrained (https://nlp.stanford.edu/projects/glove/)
# data table is faster than read_delim
glove_wts <- data.table::fread("data/glove.6B.300d.txt", quote = "", data.table = FALSE) %>% 
  as_tibble()

# convert o matrix
glove_matrix <- as.matrix(glove_wts %>% select(-V1))

# add names
rownames(glove_matrix) <- glove_wts$V1

# check object
head(glove_matrix)
dim(glove_matrix)
glove_matrix["war", ]

# function to compute nearest neighbors
nearest_words <- function(word_vectors, word, n){
  selected_vector = word_vectors[word,]
  
  mult = as.data.frame(word_vectors %*% selected_vector) # dot product in R
  mult %>%
    rownames_to_column() %>%
    rename(word = rowname,
           similarity = V1) %>%
    anti_join(get_stopwords(language = "en")) %>%
    arrange(-similarity) %>%
    slice(1: n)
  
}

# e.g. 

# state
nearest_words(word_vectors, "state", n=10)
nearest_words(glove_matrix, "state", n=10)

# state
nearest_words(word_vectors, "welfare", n=10)
nearest_words(glove_matrix, "welfare", n=10)


# obama: here it gets more distinct
nearest_words(word_vectors, "obama", n=10)
nearest_words(glove_matrix, "obama", n=10)


# abortion. Here as well. 
nearest_words(word_vectors, "abortion", n=10)
nearest_words(glove_matrix, "abortion", n=10)

# see how we can do mathematical operation with wors
berlin <- glove_matrix["paris", , drop = FALSE] -
  glove_matrix["france", , drop = FALSE] +
  glove_matrix["germany", , drop = FALSE]

library("quanteda.textstats")
cos_sim <- textstat_simil(x = as.dfm(glove_matrix), y = as.dfm(berlin),
                          method = "cosine")

# cool no?
head(sort(cos_sim[, 1], decreasing = TRUE), 5)


# 2.2 - Word2Vec -------------------

# to implement the word2vec algorithm, you can use the word2vec package in R. 

# read more here: https://github.com/bnosac/word2vec

# if you actually want to build your full neural network, you can check this blog post here: 

# https://cbail.github.io/textasdata/word2vec/rmarkdown/word2vec.html

#remotes::install_github("bnosac/word2vec")

# Python provides a beautiful package to estimate word2vec called gensim. I am actually calling 
# this package to use it in R

# get data
cr = read_rds("data/congresional_records.rds")

# let's get only one session
cr_111 <- cr %>% filter(session_id==111)

# convert to data format for gensim
text <- cr_111$speech
text <- unique(text) # remove duplicates

# prepare for gensim model
text <- (str_split(text, " "))


# activate gensim
library(reticulate)
gensim <- import("gensim") # import the gensim library
Word2Vec <- gensim$models$Word2Vec # Extract the Word2Vec model
multiprocessing <- import("multiprocessing") # For parallel processing

# seeting parameters
WINDOW_SIZE <- as.integer(6)  # how many words to consider left and right
DIM <- as.integer(300) # dimension of the embedding vector
INIT <- as.integer(1310)
NEGATIVE_SAMPLES <- as.integer(1)  # number of negative examples to sample for each word
EPOCHS <- 5L
MIN_COUNT <- 10L
WORKERS <- as.integer(as.integer(RcppParallel::defaultNumThreads()))
print(WORKERS)

## run the model. it is actually quite fast using gensim
start_time_est <- Sys.time()
basemodel = Word2Vec(text, 
                     workers = WORKERS,
                     vector_size = DIM, 
                     window = WINDOW_SIZE, 
                     sg = 1L,
                     min_count = MIN_COUNT,
                     epochs = EPOCHS,
                     compute_loss = TRUE
                     )
Sys.time() - start_time_est

# convert to a matrix
library(Matrix)
embeds <- basemodel$wv$vectors
rownames(embeds) <- basemodel$wv$index_to_key

# save
saveRDS(embeds, file ="data/word2vec_gensim.rds")


# again see nearest words                      
nearest_words(embeds, "obama", n=10)

# 2.2 - Pre-Trained Word2Vec -------------------
gensim_downloader <- import("gensim.downloader") # import the gensim downloader

# this will take a long time to run because it has 3000000 parameters
#wv = gensim_downloader$load('word2vec-google-news-300')


#  3 - Applications --------------------------------------------------------

# Beyond this type of exploratory analysis, word embeddings can be very useful in analyses of large-scale text corpora. 
# Let's showcase how we can use word embeddings in two different ways: 

# - to expand existing dictionaries; 
# - ans as a way to build features for a supervised learning classifier. 


#  3.1 - Expanding Dictionaries --------------------------------------------------------

# Let's work with the pre-trained glove model we loaded earlier

# glove matrix
dim(glove_matrix)

# let's load a quanteda dictionary: 
library(quanteda.dictionaries)
quanteda.dictionaries::data_dictionary_MFD
pos.words <- data_dictionary_LSD2015[['positive']]
neg.words <- data_dictionary_LSD2015[['negative']]

pos.words[[1]]
neg.words[[2]]

# anything we want to add?
nearest_words(glove_matrix,"ability", 10)
nearest_words(glove_matrix,"abandon", 10)

# let's see what it would take for us to replicate Gennaro's paper
care_virtue = quanteda.dictionaries::data_dictionary_MFD["care.virtue"][[1]]
care_vice = quanteda.dictionaries::data_dictionary_MFD["care.vice"][[1]]

# convert to a dataframe
dict = bind_rows(tibble(dictionary="care_vice", 
                 words=care_vice), 
          tibble(dictionary="care_virtue", 
                  words=care_virtue))

# convert to a matrix
glove_df <- glove_matrix %>%
  as_tibble() %>%
  mutate(words=rownames(glove_matrix))

# join
dict <- left_join(dict, glove_df)

# What is the word vector for each dictionary?
dict = dict %>%
  group_by(dictionary) %>%
  summarise(across(contains("V"), ~mean(.x, na.rm=TRUE)))

# then you need to repeat the same process for your documents and calculate some similarity metric to your new dictionary
dict

# 3.2 - In a supervised Learning Task ---------------------------------------------------

## Lets first start with a simple lasso classifier, similar to what we saw before in class. 
## This example is borrowed from material from Pablo Barbera. 

## Let's work with a data on incivility on facebook comments
## get the dataset here: https://www.dropbox.com/scl/fi/0v982b9knatid7r8gfluz/incivility.csv?rlkey=x4kv3huqzg4yll4q7czyd8ieq&dl=0


fb <- read_csv("data/incivility.csv")


# create a dfm + some pre-processing
fbdfm <- fb %>% 
  corpus(text_field="comment_message") %>% 
  tokens(remove_url=TRUE) %>% 
  dfm() %>% 
  dfm_wordstem() %>% 
  dfm_remove(stopwords("english")) %>% 
  dfm_trim(min_docfreq = 2, verbose=TRUE)

# let's run a bag-of-words lasso classifier:
set.seed(777)
training <- sample(1:nrow(fb), floor(.80 * nrow(fb)))
test <- (1:nrow(fb))[1:nrow(fb) %in% training == FALSE]

# estimate lasso
library(glmnet)
lasso <- cv.glmnet(fbdfm[training,], fb$attacks[training], 
                   family="binomial", alpha=1, nfolds=5, intercept=TRUE)

# get accuracy
preds <- predict(lasso, fbdfm[test,], type="class")

# confusion matrix
library(caret)
tabclass_lasso <- table(preds, fb$attacks[test])
lasso_cmat<-confusionMatrix(tabclass_lasso, mode = "everything", positive = "1")

# function to extract main measures
print_cmat <- function(confusion_matrix){
  message("Positive Class: ", confusion_matrix$positive, "\n",
          "Accuracy = ", round(confusion_matrix$overall["Accuracy"], 2), "\n",
          "Precision = ", round(confusion_matrix$byClass["Precision"], 2), "\n",
          "Recall = ", round(confusion_matrix$byClass["Recall"], 2))
}

# print
print_cmat(lasso_cmat)

## Let's add the embeddings


# keeping only embeddings for words in corpus
fbdfm <- fb %>% 
  corpus(text_field="comment_message") %>% 
  tokens(remove_url=TRUE, remove_punct=TRUE) %>% 
  dfm() %>% 
  dfm_remove(stopwords("english")) # notice no steaming here

# keep only the features I have in my data
w2v <- glove_matrix[rownames(glove_matrix) %in% featnames(fbdfm),]

# let's do one comment as an example
fb$comment_message[3] # raw text

# bag-of-words DFM
vec <- as.numeric(fbdfm[3,])

# which words are not 0s?
(doc_words <- featnames(fbdfm)[vec>0])

# let's extract the embeddings for those words
embed_vec <- w2v[rownames(w2v) %in% doc_words, ]

# and now we aggregate to the comment level
embed <- colMeans(embed_vec)

# instead of feature counts, now this is how we represent the comment:
round(embed,2)

## now the same thing but for all comments:
# creating new feature matrix for embeddings
embed <- matrix(NA, nrow=ndoc(fbdfm), ncol=300)

for (i in 1:ndoc(fbdfm)){
  if (i %% 100 == 0) message(i, '/', ndoc(fbdfm))
  # extract word counts
  vec <- as.numeric(fbdfm[i,])
  # keep words with counts of 1 or more
  doc_words <- featnames(fbdfm)[vec>0]
  
  # extract embeddings for those words
  embed_vec <- w2v[rownames(w2v) %in% doc_words, ]
  
  # Single word
  if(length(embed_vec)==300){
  embed[i,] <- embed_vec
  } else{
  embed[i,] <- colMeans(embed_vec, na.rm=TRUE)
  }
}


# check the difference in the matrices dimensions
dim(embed)
dim(fbdfm)


# there are some rows with NA's
rows_with_na <- apply(embed, 1, function(row) any(is.na(row)))
which(rows_with_na==TRUE)

# remove these
embed_ <- embed[-which(rows_with_na==TRUE), ]
dim(embed_)

# same for outcomes
fb_ <- fb[-which(rows_with_na==TRUE),]

# get training and test
set.seed(777)
training <- sample(1:nrow(fb_), floor(.80 * nrow(fb_)))
test <- (1:nrow(fb_))[1:nrow(fb_) %in% training == FALSE]

# estimate lasso
lasso <- cv.glmnet(embed_[training,], fb_$attacks[training], 
                   family="binomial", alpha=1, nfolds=5, parallel=TRUE, intercept=TRUE)


# computing predicted values
preds <- predict(lasso, embed_[test,], type="class")


# get accuracy
preds <- predict(lasso,  embed_[test,], type="class")

# confusion matrix
library(caret)
tabclass_lasso <- table(preds, fb_$attacks[test])
lasso_cmat_wv<-confusionMatrix(tabclass_lasso, mode = "everything", positive = "1")

# print
print_cmat(lasso_cmat)
print_cmat(lasso_cmat_wv)

# Better results on accuracy and precision, but not as good as in recall, all of this with a much denser matrix. 
