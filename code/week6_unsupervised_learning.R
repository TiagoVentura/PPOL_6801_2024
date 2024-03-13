##############################################################################
# File-Name: week6_unsupervised_learning.r
# Date: February 27, 2024
# author: Tiago Ventura
# course: PPOL 6801 - text as data
# topics: unsupervised learning
# Machine: MacOS High Sierra
##############################################################################

# 0 -  Introduction -------------------------------------------------------------

## This script covers topics related to using unsupervised learning models for text. We will cover 
## simple clustering methods, and then a variety of flavors from topic models

# loading packages

pacman::p_load(tidyverse, quanteda, quanteda.corpora, quanteda.textstats, quanteda.textmodels, 
               rjson, here)


# 1 - Dataset ----------------------------------------------------
# We will work with a news dataset from HuffPost available at Kaggle
# download here: "https://www.kaggle.com/datasets/rmisra/news-category-dataset#News_Category_Dataset_v2.json"

# create json
json_file="data/news.json"

# readlines
temp = readLines(json_file)

# open by entry (some entries were corrupted)
news_data <- map(temp, ~ fromJSON(.x) %>% as_tibble()) %>% bind_rows() 

# see the data
news_data

# 2 - Clustering --------------------------------------------------

# let's limit to a smaller set of topics
names_ = names(sort(table(news_data$category), decreasing = TRUE)[1:10])

news_data_ <- news_data %>% filter(category%in%names_) %>% sample_n(10000)

# little pre-processing
dfm_news <-  corpus(news_data_, text_field = "headline") %>%
             tokens(remove_punct = TRUE, remove_numbers = TRUE)  %>%
              dfm() %>% 
              dfm_wordstem() %>% 
              dfm_remove(stopwords("en")) 

# normalize the text ~ matter if we are using euclidean distance for kmeans
dfm_news <-  dfm_weight(dfm_news, scheme = "prop") 

# kmeans
n.clust<- 10
set.seed(1310) 

# estimate clusterd
k_cluster<- kmeans(dfm_news, centers = n.clust)

# see output
str(k_cluster, 2) 

## clusters for document assignment
k_cluster$cluster

## center for words, row=cluster, columns=words
dim(k_cluster$centers)
k_cluster$centers[1,]

# assignment:
table(k_cluster$cluster)
table(news_data_$category) # not great, right?

# labeling the clusters

##just use the ``biggest" in each category
key_words<- matrix(NA, nrow=n.clust, ncol=10)

for(z in 1:n.clust){
  key_words[z,]<- colnames(dfm_news)[order(k_cluster$center[z,], decreasing=T)[1:10]]
}

# see as a tibble
key_words_bind <- key_words %>% 
  as_tibble() %>%
  pivot_longer(cols = contains("V"),
               names_to = "cluster", 
               values_to = "words") %>%
  group_by(cluster) %>%
  summarise(words= list(words)) %>%
  mutate(words = map(words, paste, collapse = ", ")) %>%
  unnest() 

# merge clusters with text
clusters <- tibble(text=names(k_cluster$cluster),
          cluster=as.character(k_cluster$cluster))

# bind all
df_clusters <- bind_cols(as_tibble(docvars(dfm_news)), clusters)

# get text
df_clusters$text <- news_data_$headline

# see some cases from each cluster
df_clusters %>% 
  group_by(cluster) %>%
  sample_n(5) %>%
  select(cluster, text) %>% 
  View()

# if you look at it, it is not soo bad! 
# But the deterministic approach is not ideal for dealing with complex text

# Topic Models ------------------------------------------------------------

# Let's start with topic models. Let's first see a bit about the traditional LDA model. 

# here is a really nice example of using  LDA in R
# https://www.tidytextmining.com/topicmodeling

# call package
library(topicmodels)


# pre-processing
dfm_news <-  corpus(news_data %>% mutate(id=1:nrow(.)),
                    docid_field="id", text_field = "headline") %>%
  tokens(remove_punct = TRUE, remove_numbers = TRUE, remove_symbols=TRUE)  %>%
  dfm() %>% 
  dfm_remove(stopwords("en")) %>%
  dfm_trim(min_docfreq = 2,
           verbose = TRUE) 

# add healdine back as variables
dfm_news$text <- news_data$headline

  
# keep only the rows with non-zero
dfm_news_non_na <-  dfm_news[rowSums(dfm_news) > 0,]

# check if things are working
dim(dfm_news_non_na)
dim(docvars(dfm_news_non_na))
colnames(docvars(dfm_news_non_na))
length(docid(dfm_news_non_na))

# estimate LDA with K topics
length(table(news_data$category))

# set the number of topics
K <- 30


# estimate
lda <- LDA(dfm_news_non_na, 
           k = K, 
           method = "Gibbs",
           control = list(verbose=25L, seed = 123,
                          burnin = 100, iter = 500))

# We can use `get_terms` to the top `n` terms from the topic model, 
terms <- get_terms(lda, 10)

# terms is a matrix. Easy to convert to a df and clean it a bit
terms_df <- as_tibble(terms) %>%
  janitor::clean_names() %>%
  pivot_longer(cols = contains("topic"), 
               names_to = "topic", 
               values_to = "words") %>%
  group_by(topic) %>%
  summarise(words= list(words)) %>%
  mutate(words = map(words, paste, collapse = ", ")) %>%
  unnest() 

terms_df  

# get_topics to predict the top `k` topic for each document. 
# This will help us interpret the results of the model.
topics <- get_topics(lda, 1)


# the tidy text package gives you a nice way to look at this
library(tidytext)

# two main parameters

# word-topic probabilities ~ beta parameters
beta_lda <- tidy(lda, matrix="beta")

beta_lda %>% 
  arrange(topic, desc(beta)) 


# document-topic probabilities ~ gamma parameters
gamma_lda <- tidy(lda, matrix = "gamma")

# easy to find top terms for topics
beta_lda_top_terms <- beta_lda %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

beta_lda_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

# Easy to get topics for each document
main_doc_topic <- gamma_lda %>%
  group_by(topic) %>%
  slice_max(gamma) 

# merge back

# recreate the dfm
df <- as_tibble(docvars(dfm_news_non_na)) %>%
       mutate(document=docid(dfm_news_non_na)) 
  
# merge main doc and top terms
df <- df %>%
  right_join(main_doc_topic) %>%
  select(topic, category, short_description, text) %>%
  mutate(topic=paste0("topic_", topic)) %>%
  left_join(terms_df)

# lot of information you can get here!
df  %>% select(topic, words, text, short_description) %>% slice(1:10) %>% View()

# 3 - STM -----------------------------------------------------------------

# the stm is one of my favorites packages to run topic models. not because of the possibility 
# of adding covariates to the model, but because I think the functions are very intuitive to run
# and the results of the model are very stable

# read more here: https://www.structuraltopicmodel.com/

# let's see some examples of how to use stm
library(stm)

# the first step is to use quanteda to convert the dfm to a format stm can read

dfm_stm <-  quanteda::convert(dfm_news_non_na, to = "stm")

# see what this looks like
# matrix with id for words and counts
dfm_stm$documents[[1]]

# Models 
stm_m <- stm(documents=dfm_stm$documents,
              vocab=dfm_stm$vocab, 
              data = dfm_stm$meta, 
              K = 30,
              init.type = "Spectral")

#save(stm_m, file="data/stm_model.rdata")
# download here: https://www.dropbox.com/scl/fi/29qmppum357oa0snnejyt/stm_model.rdata?rlkey=yjf548fvd6z4eoz0dmos3ezil&dl=0

load("data/stm_model.rdata")
# STM offers a series of pre-built functions to analyze results. 
# Let's see some of them

# Check topic
?labelTopics

labelTopics(stm_m)

#  topic proportion
plot(stm_m)

# finding documents
thoughts3 <- findThoughts(stm_m, texts = dfm_stm$meta$text,
                          n = 2, topics = 14)$docs[[1]]

# see main words
labelTopics(stm_m, topics=14)
plotQuote(thoughts3, width = 30, main = "Topic 14")

# Topic correlation
mod.out.corr <- topicCorr(stm_m)$cor

# Let's now try a more tidy approach to this.

# Topics and words
td_beta <- tidy(stm_m)
td_gamma <- tidy(stm_m, matrix = "gamma",
                 document_names = rownames(dfm_stm))


## Top Terms
top_terms <- td_beta %>%
  arrange(beta) %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  arrange(-beta) %>%
  select(topic, term) %>%
  summarise(terms = list(term)) %>%
  mutate(terms = map(terms, paste, collapse = ", ")) %>%
  unnest()

# gamma proportions + terms
gamma_terms <- td_gamma %>%
  group_by(topic) %>%
  summarise(gamma = mean(gamma)) %>%
  arrange(desc(gamma)) %>%
  left_join(top_terms, by = "topic") %>%
  mutate(topic_ = paste0("Topic ", topic),
         topic_ = reorder(topic_, gamma))


# Looking at specific stopics
library(scales)
gamma_terms %>%
  # pick any topic that makes sense for you
  slice(1:10) %>%
  ggplot(aes(topic_, gamma, label = terms, fill = topic_)) +
  geom_col(show.legend = FALSE) +
  geom_text(hjust = 0, nudge_y = 0.0005, size = 4)+
  coord_flip() +
  scale_y_continuous(expand = c(0,0),
                     limits = c(0, .1),
                     labels = percent_format()) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(size = 22, face="bold"),
        plot.subtitle = element_text(size = 13),
        axis.title.x  = element_text(size=12, face="italic", hjust=1),
        axis.text.y =  element_text(size=12, face="bold"),
        axis.ticks = element_blank(),
        panel.grid = element_blank()) +
  labs(x = NULL, y = "Topic Proportions with most associated words")


# Main documents in each topic
topics_top_speeches <- td_gamma %>%
  group_by(topic) %>%
  top_n(30) %>%
  arrange(document) %>%
  left_join(top_terms, by = "topic") %>%
  arrange(topic, desc(gamma)) 

# merge with complete text
sp_ <- as_tibble(docvars(dfm_news_non_na)) %>%
  mutate(document=as.integer(docid(dfm_news_non_na)))

meta <- topics_top_speeches %>%
  left_join(sp_)

## Examples: Topic
meta %>% 
  filter(topic==3) %>%
  arrange(desc(gamma)) %>%
  select(topic, gamma,  terms, text) %>%
  slice(1:2) 

# not bad!

# Main topic in each documents
speeches_topic <- td_gamma %>%
  group_by(document) %>%
  top_n(1) %>%
  arrange(document) %>%
  left_join(top_terms, by = "topic") %>%
  arrange(document, desc(gamma)) %>%
  left_join(sp_)

speeches_topic

# Selecting the number of topics ------------------------------------------

# How many Topics  -------------------------------------------------------------------------
## Deciding the topics using data-driven approach
library(stm)
library(ggrepel)
seq(10, 210, 20)

# run many models
# this will take a long time.
many_models_search_k <- searchK(dfm_stm$documents, dfm_stm$vocab, K = seq(10, 210, 20),
                                data = dfm_stm$meta, init.type = "Spectral")

# load pre-saved results
load(here("data", "many_models_stm.Rdata"))
# download here: https://www.dropbox.com/scl/fi/f1qdjx44cgzzwchobo28q/many_models_stm.Rdata?rlkey=slin76qp1233dcd6ng9cymsww&dl=0


# compare exclusivity vs coherence
many_models_search_k$results %>% 
  select(K, exclus, semcoh) %>% unnest() %>%
  ggplot(aes(y=exclus, x=semcoh, label=K)) +
  geom_point(size = 2, alpha = 0.7) +
  geom_smooth() +
  geom_text_repel() +
  theme_minimal() +
  labs(x = "Semantic coherence",
       y = "Exclusivity",
       title = "Comparing exclusivity and semantic coherence")


# see all metrics
many_models_search_k$results %>%
  gather(Metric, Value, -K) %>% unnest() %>%
  ggplot(aes(K, Value)) +
  geom_smooth(size = 1.5, alpha = 0.7, show.legend = FALSE, color="tomato2") +
  facet_wrap(~Metric, scales = "free_y") +
  labs(x = "K (number of topics)",
       y = NULL,
       title = "Model diagnostics by number of topics",
       subtitle = "") + 
  theme_minimal()

# More materials: 

# some other sources you can use to learn about topic models

# for keyATM: https://keyatm.github.io/keyATM/
# Bert Based Topic Models: https://maartengr.github.io/BERTopic/index.html
# tidytext https://www.tidytextmining.com/topicmodeling.html
# in pythonn: https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05
# human validation of topic models: https://dl.acm.org/citation.cfm?id=2984126


## have fun!



