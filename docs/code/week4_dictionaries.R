##############################################################################
# File-Name: week3_descriptive_inference.r
# Date: January 31, 2024
# author: Tiago Ventura
# course: PPOL 6801 - text as data
# topics: dictionaries and off-the-shelf toxicity models
# Machine: MacOS High Sierra
##############################################################################

# 0 -  Introduction -------------------------------------------------------------

## This script provides covers topics related toÇ 

## 1- dictionaries for supervised classification

## 2- Using off-the-shelf Models for classification taskss

# loading packages
pacman::p_load(tidyverse, quanteda, quanteda.corpora, quanteda.textstats)

# 1 - Dictionaries -------------------------------------------------

# As we saw in class, one of the most common applications of 
# dictionary methods is for sentiment classification. Let's see a few
# options to do sentiment classification in R

# today, we will work with the streamming chats comments from my article that we read in class. 

## download here: https://www.dropbox.com/scl/fi/hodfhmfkuf80skutg8qo0/comments_facebook_presidental_elections.csv?rlkey=q5cw84hfp9sx3ordxrf5261ej&dl=0

d <- read_csv("data/comments_facebook_presidental_elections.csv")

# see the data
glimpse(d)

# create an id
d <- d %>% rowid_to_column()


# 1.1 Sentiment Classification - Quanteda -------------------------------------------------

# Quanteda provides easy access to a variety of dictionary methods. 
# To do so, we need to install quanteda.dictionaries and quanteda.sentiments first

# install
#devtools::install_github("kbenoit/quanteda.dictionaries")
#devtools::install_github("quanteda/quanteda.sentiment")

# call
pacman::p_load("quanteda.dictionaries", "quanteda.sentiment")


# Let's start working with the Lexicoder Sentiment Dictionary (from Young and Soroka)

# load the dictionary
data(data_dictionary_LSD2015)

# notice here: it loads a class dictionary. This is not a dictionary class
# as you have in python. Those do not exist in R! This is just a special 
# data structure created by quanteda

# let's see what this structure looks like
str(data_dictionary_LSD2015, 2)

# let'see some examples here
pos.words <- data_dictionary_LSD2015[['positive']]
neg.words <- data_dictionary_LSD2015[['negative']]

# a look at a random sample of positive and negative words
sample(pos.words, 10)
sample(neg.words, 10)

# or
print(data_dictionary_LSD2015)

# let's convert our text to a dfm
d_corpus <- d %>%
  corpus(text_field = "comments") %>%
  tokens() %>%
  dfm()

# There are a few different ways for you to apply the dictionary in quanteda

# 1.1.1 textstats_polarity -----------------------------------

# let's undertand the function first
# look at fun argument = it outputs the log of pos/neg words
?textstat_polarity

# apply
lsd_results <- d_corpus %>% 
  textstat_polarity(dictionary = data_dictionary_LSD2015)

# see outputs
lsd_results

# let's merge back with the original data

# get vars back
df_sent <- docvars(d_corpus) %>% mutate(doc_id=docnames(d_corpus)) %>% 
  left_join(lsd_results) %>%
  glimpse()

# Plot sentiment by debate
df_sent %>%
  group_by(debate) %>%
  summarise(m=mean(sentiment, na.rm=TRUE))


# 1.1.2 DFM Lookup -----------------------------------

# another way is to just do a lookup on the terms

# select only the "negative" and "positive" categories
data_dictionary_LSD2015_pos_neg <- data_dictionary_LSD2015[1:2]

# use tokens lookup

toks_gov_lsd <- d %>%
                corpus(text_field = "comments")  %>% 
                tokens() %>%
                tokens_lookup(dictionary = data_dictionary_LSD2015_pos_neg)

# create a document document-feature matrix and group it by day
dfmat_gov_lsd <- dfm(toks_gov_lsd)
dfmat_gov_lsd

# we can easily merge back again
d$positive = as.numeric(dfmat_gov_lsd[, 2])
d$negative = as.numeric(dfmat_gov_lsd[, 1])

# See dataframe
d %>% glimpse()


## we can with this dataset answer some easy descriptive questions

# get a sentiment score for the entire data
d <- d %>% 
  mutate(score=positive-negative)

# what is the average sentiment score?
mean(d$score)

# what is the top-5 positive and most negative comment?

# positive
d %>% 
  select(comments, score) %>% 
  arrange(desc(score)) %>%
  slice(1:5)

# negative
d %>% 
  select(comments, score) %>% 
  arrange(score) %>%
  slice(1) %>% pull(comments) 

# what is the proportion of positive, neutral, and negative tweets?
d <- d %>%
  mutate(score_label=case_when(score>0 ~ "Positive",
                               score<0 ~ "Negative"))
# see proportions
d %>%
  janitor::tabyl(score_label)


# 1.2 Vader Sentiment Classification -------------------------------------------------

# we saw in-class the Vader dictionary, that is super useful when dealing
# with social media data

# nltk in python has a simple implementation of the vader dictionary. 

# in R, you need to use a specific package to do so. 

# installation
install.packages("vader")
library(vader)

# main function: get_vader
get_vader("uau! awesome package")

# The result is an R vector with named entries:

# - word_scores: a string that contains an ordered list with the matched scores for each of the words
# - compound: the final valence compound of VADER for the whole text after applying modifiers and aggregation rules
# - pos, neg, and neu: the parts of the compound for positive, negative, and neutral content. 

# write a function to tidy the results
get_vader_tidy <- function(text){
get_vader(text) %>% 
 tibble(outcomes=names(.), 
        values=.)
}  

# apply to text from comments
d <- read_csv("data/comments_facebook_presidental_elections.csv")

# create a list
vader_outputs = list()

# loop through
for(i in 1:1000){
vader_outputs[[i]] =  get_vader_tidy(d$comments[[i]]) 
print(i)
}
toc()


# unnest
d$vader_output <- vader_outputs

# unnest
d %>% unnest(vader_output) %>% filter(outcomes=="compound") %>%
  select(comments, outcomes, values)

# 3 - Perspective API -------------------------------------------------------------------------

# Perspective is a free API that uses machine learning to identify "" comments, 
# making it easier to host better conversations online.

# It was developed by Jigsaw, and later bought by google. It is used by 
# some major news outlets as a content moderation tool. 

# read more about it here: https://www.perspectiveapi.com/

# I have used the Perspective API to measure online toxicity in
# three peer-reviewed papers:

## https://journalqd.org/article/view/2573

## https://academic.oup.com/joc/article-abstract/71/6/947/6415947

## https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0281475


## Google provides an API that allows you to query the model and get results


## 3.1 Open the data -------------------------------------------------------------------------------------

## We are actually working here with the streamming chats comments
## from my article that we read in class. 

## download here: https://www.dropbox.com/scl/fi/hodfhmfkuf80skutg8qo0/comments_facebook_presidental_elections.csv?rlkey=q5cw84hfp9sx3ordxrf5261ej&dl=0

d <- read_csv("data/comments_facebook_presidental_elections.csv")

d_for_tox <- d %>%
  rowid_to_column(var='unique_id') %>%
  drop_na(comments)


## 3.2 Query the API -------------------------------------------------------------------------------------

# install wrapper
#devtools::install_github("favstats/peRspective")

# call the package
library(peRspective)

# Read here how to get an api key: https://developers.perspectiveapi.com/s/docs-get-started?language=en_US
#usethis::edit_r_environ()

#
# Query the api
outputs <- prsp_stream(d_for_tox %>% slice(1:10), text=comments, text_id=unique_id,
                          score_model =  peRspective::prsp_models,
                          safe_output = T, verbose = T)

outputs

# load data with all the comments
toxicity <- read_csv("data/outputs_tox.csv")

# make it long

toxicity <- toxicity %>%
  pivot_longer(cols=-c("text_id", "error"),
               names_to="variables",
               values_to="scores")


# Merge

toxicity <- left_join(toxicity, d_for_tox, by=c("text_id"="unique_id"))



## Clean the data

toxicity <- toxicity %>%
  mutate(label=str_replace_all(debate, "_", " "),
         label=str_to_title(label),
         label=str_trim(str_remove_all(label, "_|Manual"))) %>%
  mutate(label=str_replace_all(label, "Abc", "ABC"),
         label=str_replace_all(label, "Nbc", "NBC"),
         label=str_replace_all(label, "Fox", "FOX")) %>%
  arrange(label) %>%
  mutate(label=forcats::fct_inorder(label))  %>%
  mutate(Attribute=str_replace_all(str_to_title(variables), "_", " "))


# Remove Some Attributes

types_tox <- unique(toxicity$variables)
types_tox_red <-  c("TOXICITY", "SEVERE_TOXICITY",
                    "INSULT", "THREAT")

tox_red <- toxicity %>%
  filter(variables%in% types_tox_red) %>%
  mutate(Attribute=fct_relevel(Attribute, "Toxicity", "Severe toxicity", "Threat", "Insult"))

# Visualize the results 

library(rebus)
library(wesanderson)

av <- tox_red %>%
  group_by(label, Attribute) %>%
  summarise(m=mean(scores, na.rm = TRUE)) %>%
  mutate(newsource=str_sub(label, 1, 3))

pal = wes_palette("Zissou1")
buda= wes_palette("GrandBudapest1")

ggplot(av, aes(y=m, x=fct_rev(label), fill=fct_rev(Attribute))) +
  geom_col(color="black", width=0.7, alpha=.6) +
  theme_minimal(base_size = 14) +
  facet_grid(newsource~ Attribute, scales = "free_y") +
  ylim(0,.4) +
  coord_flip() +
  scale_fill_manual(values=c(pal[1], pal[3], buda[3], pal[5])) +
  guides(fill=FALSE) +
  theme(axis.title.x = element_text(hjust=1)) +
  labs(y="Average Scores", x="") +
  theme(plot.margin = margin(1, 1, 1, 1, "cm"))



