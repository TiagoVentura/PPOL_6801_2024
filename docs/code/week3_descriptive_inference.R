##############################################################################
# File-Name: week3_descriptive_inference.r
# Date: January 31, 2024
# author: Tiago Ventura
# course: PPOL 6801 - text as data
# topics: document similarity,tf-idf, and readability measures in text
# Machine: MacOS High Sierra
##############################################################################


# 0 -  Introduction -------------------------------------------------------------

## This script provides covers topics related to descriptive inference using text. We cover

## Comparing documents using cosine similarity
## TF-IDF
## Measures of readability and lexical diversity

pacman::p_load(tidyverse, quanteda, quanteda.corpora, quanteda.textstats)

# 1 - Comparing Documents -----------------------------------------------------

# Let's first ilustrate how to calculate consine similarity and euclidean distance

# user-defined function to calculate the cosine similarity of a function
calculate_euclidean_distance <- function(vec1, vec2){
ec_distance <- sqrt(sum((x - y)^2))
return(ec_distance)
}

# cosine similarity
calculate_cosine_similarity <- function(vec1, vec2=b) { 
  nominator <- vec1 %*% vec2  # element-wise multiplication
  denominator <- sqrt(vec1 %*% vec1)*sqrt(vec2 %*% vec2)
  return(nominator/denominator)
}

# three dimensional example
x <- c(1, 2, 3)
y <- c(1, 2, 3)

# Same vectors = distance should be zero,  similarity should be 1
calculate_cosine_similarity(x, y)
calculate_euclidean_distance(x, y)

# example 2
a <- c(1, 2, 3)
b <- c(-1, -2, -3)

# what should we get?
calculate_cosine_similarity(a, b)

# Can we ever observe a vector like b in a DFM?

# example 3
a <- c(1, 2, 3, 0)
b <- c(0, 0, 0, 1)

# what should we get?
calculate_cosine_similarity(a, b)

# 1.1 Similarity Measures on Quanteda ----------

# Let's know use these similarity measures with real text and using Quanteda

# Open the tweets dataset we worked with last week

# download the data here: https://www.dropbox.com/scl/fi/l5rc7nptc23en7fj1halz/tweets_congress.csv?rlkey=joqnuldgh3xanx8y9h8wvkf8p&dl=0

# open dataset
tweets <- read_csv("data/tweets_congress.csv")

# let me sample a few politicians
sample_n_politicians <- tweets %>%
  group_by(author) %>%
  select(author) %>%
  distinct() %>% 
  ungroup() %>%
  sample_n(100) 
  
# merge back the full data
tweets_g = left_join(sample_n_politicians, tweets)

# let me aggregate politicians tweets
tweets_g <- tweets_g %>%
            group_by(author) %>%
            summarize(text=paste0(text, collapse = " ")) %>% 
            ungroup()

# convert to a corpus
tweets_corpus  <- corpus(tweets_g, text_field="text")

# pre-process and convert to dfm
tweets_dfm <- tokens(tweets_corpus,
       remove_punct = TRUE, 
       remove_numbers= TRUE, verbose = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_wordstem() %>%
  dfm() %>%
  dfm_trim(min_docfreq = 0.05,
             max_docfreq = 0.95,
             docfreq_type ="prop", 
             verbose = TRUE)

# aggregate at groups
dfm_grouped <- dfm_group(tweets_dfm, author)
print(dfm_grouped)


# check similarity
?textstat_simil()
simil_tweets <- textstat_simil(dfm_grouped,  
               margin = "documents",
               method = "cosine")
# convert to a matrix
as.matrix(simil_tweets)

# names
rownames(as.matrix(simil_tweets))

# convert to a tibble
df_sim = as_tibble(as.matrix(simil_tweets)) %>% 
  mutate(names = rownames(as.matrix(simil_tweets))) %>%
  select(names, everything()) 

# see least similar
df_sim %>%
  filter(names=="SenTedCruz") %>%
  pivot_longer(cols = -c(names), 
               names_to = "rep", 
               values_to="similarity") %>%
  arrange(similarity) 
  
# easily plot this as a heat map. Let's use a random sample of 10 politicians
df_sim_sample = df_sim %>%
  sample_n(10) 

# get the same 10 in columnes
df_sim_samples <- df_sim_sample %>% 
                    select(names, df_sim_sample %>% pull(names)) %>% 
                   pivot_longer(cols = -c(names), 
                      names_to = "rep", 
                      values_to="similarity")

# heatmap
ggplot(df_sim_samples,
       aes(x=names,y=rep, 
           fill=similarity))+
  geom_tile(colour="gray95",size=0.5, alpha=.8)  +
  geom_text(aes(label = round(similarity, 2)), color = "black", size = 4) +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal() +
  theme(axis.text.x =  element_text(angle=90, hjust=1, size=10), 
        strip.text = element_text(color = "#22211d",
                                  size = 14, face="italic"), 
        plot.caption = element_text(size=10)) 
  
# Quanteda also allows to easily extract other measures of similarity, and distance

dist_tweets_euc <- textstat_dist(dfm_grouped,  
                               margin = "documents",
                               method = "euclidean")

# what does these distances mean? is 1042 a lot? a little? hard to scale!
as.matrix(dist_tweets_euc)


# 2 - Weighting TF-iDF -----------------------------------------------------

# Quanteda offers a easy way to compute tf-idf weights.
# Let's re-run the cosine similarity analysis using tf-idf and compare results


# convert to tfidf
tfidf_tweets <- tweets_dfm %>% 
                dfm_tfidf(scheme_tf = "prop") %>% 
                dfm_group(author, force = TRUE) 
dfm_grouped
tfidf_tweets

#similarity 
simil_tweets <- textstat_simil(tfidf_tweets,  
                               margin = "documents",
                               method = "cosine")
# convert to a matrix
as.matrix(simil_tweets)

# names
rownames(as.matrix(simil_tweets))

# convert to a tibble
tfidf_sim = as_tibble(as.matrix(simil_tweets)) %>% 
  mutate(names = rownames(as.matrix(simil_tweets))) %>%
  select(names, everything()) 

# get the 10 closest to ted cruz
top10_df = df_sim %>%
  filter(names=="SenTedCruz") %>%
  pivot_longer(cols = -c(names), 
               names_to = "rep", 
               values_to="similarity") %>%
  arrange(desc(similarity)) %>%
  slice(2:21) %>% mutate(cell="No Weight") %>% rownames_to_column()

top10_idfdf = tfidf_sim %>%
  filter(names=="SenTedCruz") %>%
  pivot_longer(cols = -c(names), 
               names_to = "rep", 
               values_to="similarity") %>%
  arrange(desc(similarity)) %>%
  slice(2:21) %>% mutate(cell="TF-IDF") %>% rownames_to_column()

# bind dataframes
cos_sim_tweets = bind_rows(top10_df, top10_idfdf) %>%
                  mutate(rowname=fct_rev(fct_inorder(rowname)))
  
  
levels(cos_sim_tweets$rowname)

# plot
ggplot(cos_sim_tweets, aes(y=rowname, x=.5, fill=cell, label=rep)) +
  geom_label() +
  facet_wrap(~cell, scales="free_y") +
  theme_minimal() +
  labs(title="Top 10 Closest Representatives to Ted Cruz", 
       subtitle = "Method: Cossine Similarity with Tweets",
       x="Cossine Similarity") +
  ylab("Ranking Similarity") +
  theme(axis.text.x = element_blank())

# 3 - Lexical Diversity ------------------------------------------------------

rm(list = ls())
# Load in data: SOTU 
library(quanteda.corpora)

# TTR (by hand) ####################
sotu_dfm <- tokens(data_corpus_sotu, 
                     remove_punct = TRUE) %>% dfm()

# Num tokens per document
num_tokens <- ntoken(sotu_dfm)

num_types <- ntype(sotu_dfm)

sotu_TTR <- num_types / num_tokens

# create a dataframe
df_ttr <- tibble(ttr=sotu_TTR, docs = names(sotu_TTR))

# Using Quanteda
# textstat_lexdiv: "calculates the lexical diversity or complexity of text(s)" using any number of measures.'
TTR <- textstat_lexdiv(sotu_dfm, 
                       measure = "TTR",
                       remove_numbers = F, remove_punct = F, remove_symbols = F)
# dataframe
df_ttr <-  tibble(TTR)

# combine with metadata
metadata = bind_cols(docvars(data_corpus_sotu), document = docnames(data_corpus_sotu))

# join
df_ttr <- left_join(df_ttr, metadata)


# calculate TTR by year
df_ttr = df_ttr %>%
          mutate(year=year(Date)) %>%
          group_by(year) %>%
          mutate(ttr_year=mean(TTR)) %>%
          group_by(party) %>%
          mutate(ttr_party=mean(TTR)) 

# see by party
df_ttr %>%
  select(party, ttr_party) %>%
  distinct() 

# see by year
df_ttr %>%
  select(year, ttr_year) %>%
  distinct() %>%
  ggplot(aes(x=year, y=ttr_year)) +
  geom_point(shape=21, fill="tomato2", color="black") +
  geom_smooth(alpha=.2, color="gray") +
  theme_minimal()


# 4 - Complexity Measures ------------------------------------------------------

# Let's compare several difference measures

all_readability_measures <- textstat_readability(data_corpus_sotu, 
                                                 c("Flesch", "Dale.Chall", 
                                                   "SMOG", "Coleman.Liau"))


# Join
all_ <- left_join(all_readability_measures, metadata)

# let's plot 
all_ <- all_ %>%
  mutate(year=year(Date)) %>%
  group_by(year) %>%
  mutate_at(c("Flesch", "Dale.Chall", 
              "SMOG", "Coleman.Liau.ECP"), ~ mean(.x, na.rm = TRUE)) %>%
  pivot_longer(c("Flesch", "Dale.Chall", 
                 "SMOG", "Coleman.Liau.ECP"), 
               names_to = "Complexity", 
               values_to = "val")

# plot
library(wesanderson)
wesanderson::wes_palette("FantasticFox1", 4)

all_ %>%
  select(year,Complexity, val) %>%
  ggplot(aes(x=year, y=val, fill=Complexity, color=Complexity)) +
  geom_smooth(alpha=.2) +
  scale_fill_manual(values=wesanderson::wes_palette("FantasticFox1", 4)) +
  scale_color_manual(values=wesanderson::wes_palette("FantasticFox1", 4)) +
  theme_minimal()

# 5 - Benoit, Munger and Spirling's measure ------------------------------------------------------

# to work with the measure developed by Benoit et al, check the github link below:

# https://github.com/kbenoit/sophistication
install.packages("spacyr")

library(spacyr)
library(reticulate) 

# tell which python I am running
Sys.setenv(RETICULATE_PYTHON ="/Users/tb186/anaconda3/envs/ppol6801/bin/python")
# tell spacy which conda environment I am using

Sys.setenv(SPACY_PYTHON = "/Users/tb186/anaconda3/envs/ppol6801/")

# install spacy
spacyr::spacy_install()
spacyr::spacy_initialize()

# install sophistication
devtools::install_github("kbenoit/sophistication")
library(sophistication)

# the package provides you with the best model from the paper
?data_BTm_bms

# if you want to go ahead with this model, and predict some new data,
# it is relatively simple
?predict_readability
sophistication::data_corpus_presdebates2016


# run this function in a new dataset
sophistication_sotu = predict_readability(data_BTm_bms, newdata = data_corpus_sotu)



