##############################################################################
# File-Name: week2_introduction_to_text_analysis.r
# Date: January 22, 2024
# author: Tiago Ventura
# course: ppol 6801 - text as data
# topics: basic introduction to text analysis using quanteda and tidy text
# Machine: MacOS High Sierra
##############################################################################


# 0 -  Introduction -------------------------------------------------------------

## This script provides a basic introduction to text analysis using primarily quanteda. Some topics we will cover:
# - loading text data in R
# - creating a corpus
# - pre processing steps: stop words, stem, lemmatization and normalization using quanteda
# - convert corpus to document feature matrices
# - basic descriptive analysis.

## `quanteda` is far from being the only package to work with text in R. Other competitors are tm, text2vec, and even just base R. 
## These packages follow a similar logic on how to process text data.

## A more distinct approach comes from the tidytext family of packages, which tries to import the tidy text philosophy (https://r4ds.had.co.nz/tidy-data.html)
## to work with text. If you are already familiar with using dplyr/tidyverse for data manipulation, the tidytext approach can be 
## super useful for descriptive tasks and pre-processing text in R. For this reason, we will cover a bit the tidytext approach 
## in the end of this script.


# 1 - Setting up your environment ---------------------------------------------

## to load packages in R, I strongly suggest you to use the `pacman` as package management tool (https://trinker.github.io/pacman/vignettes/Introduction_to_pacman.html)
## the pacman combines the install.package and library steps in a single function, basically. 

install.packages(pacman)

# Install the latest stable version of quanteda from CRAN

# the `::` allows you to access the function of a package without loading it

# install packages 
pacman::p_load(tidyverse,
               ggplot2,
               quanteda,
               tidytext) 

# 2 - Review - String Manipulation in R --------------------------------------

# To work with text, you need to know the basics of manipulating strings. 
# Since some of you have been working with Python, I will provide here a quick overview of doing string manipulation in R

# let's load some text data. We are working with a corpus of Twitter timelines for members of the congress during the 2021 election year

# download the data here: https://www.dropbox.com/scl/fi/l5rc7nptc23en7fj1halz/tweets_congress.csv?rlkey=joqnuldgh3xanx8y9h8wvkf8p&dl=0

# open dataset
tweets <- read_csv("data/tweets_congress.csv")

# where is the text? 
tweets %>% glimpse() # tweet text is stored in text variable. 

# pretty big dataset, let's get a sample
tweets_s <- tweets %>% sample_n(10000)

## 2.1 - Basic string manipulation --------------------------------------

# R stores strings as characters. 
# All basic functions you learned for dealing with characters can be used for text manipulation


# length of a string
length(tweets_s$text[[1]]) # single object

# size
nchar(tweets_s$text[1:10]) 

# mathematical functions
max(nchar(tweets_s$text[1:10])) 
min(nchar(tweets_s$text[1:10]))
mean(nchar(tweets_s$text[1:10]))

# concatenation
paste0(tweets_s$text[1:10], collapse = "\n")
paste("one", "two")
paste0("one", "two")

# comparison

# element-wise
c("trump", "biden", "who else?") == c("biden", "who else?", "trump")

# %in% for contains
c("trump", "biden", "who else?") %in% c("biden", "who else?", "trump")

## 2.2 - stringr --------------------------------------

# To more advanced string manipulation, we will use primarily the stringr package from the tidyverse family

pacman::p_load(stringr)

# see a full list for stringr function here: https://evoldyn.gitlab.io/evomics-2018/ref-sheets/R_strings.pdf/

# structure of the functions: str_<functionality>

### detection tasks --------

# to detect patterns = `str_detect()`
str_detect(tweets_s$text[1:10], "Biden")

# to count: `str_count()`
str_count(tweets_s$text[1:10], "Biden")

# to locate elements: `str_locate()`
str_locate(tweets_s$text[1:10], "Biden")

### Changing strings --------

# Replace strings: `str_replace_all()`
str_replace_all(tweets_s$text[1:10], "President Biden", "president biden")[8]

# lower case: `str_to_lower()`
# upper case: `str_to_upper()`
# title sentence: `str_to_title()`
str_to_lower(tweets_s$text[1:10])

## Extracting patterns  --------
  
# by position: `str_sub()`
str_sub(tweets_s$text[1:10], 1, 3)

# by matching `str_subset()`
str_subset(tweets_s$text[1:10], "President Biden")


# only the pattern: `str_subset()`
str_extract_all(tweets_s$text[1:10], "President Biden",  simplify = TRUE)


## 2.3 - Important Features - Tidy + Regular Expressions -------------------

### All these functions are set to work with tidy chained sequences 

# detect all tweets that mention trum
tweets_s %>%
  mutate(trump=str_detect(str_to_lower(text), "trump")) %>%
  filter(trump==TRUE)

# select only retweets
tweets_s %>%
  mutate(rt=str_sub(str_to_lower(text), 1, 3)) %>%
  filter(rt=="rt ")

### All deal with regular expression, as you likely learn in Python. 

# starts with RT
str_subset(tweets_s$text[1:100], "^RT")

#  contains "!"
str_subset(tweets_s$text[1:100], "!")

# Ending with "!"
str_subset(tweets_s$text[1:1000], "!$")

# extract handles
str_extract_all(tweets_s$text[1:1000], '@[0-9_A-Za-z]+', simplify=TRUE)

# now with hashtags...
str_extract_all(tweets_s$text[1:1000], "#(\\d|\\w)+", simplify=TRUE)

# This is not review of regular expressions. this serves only the purpose to show you how you can use it in 
# string detection in R. much more to learn online!

# see more about string manipulations here: https://r4ds.had.co.nz/strings.html

# 3 - Quanteda for Text Analysis ----------------------

# As we discussed earlier, the first step in any text analysis text is 
# to "preprocessing" the data and convert to a format (often document feature matrix) 
# before it can be passed to a statistical model. 
# we'll use the `quanteda` package for these tasks

# several quanteda tutorials are available here: https://quanteda.io/index.html

# The basic unit of work for the `quanteda` package is called a `corpus`.
# A corpus is a collection of documents and contains: text and metadata.

## 3.1 - Creating a corpus ----------------

# let's start learning how to create a corpus documents. We will do so from: 

### 3.1.1 - from a csv ----

# load tweets data again
tweets <- read_csv("data/tweets_congress.csv") %>% sample_n(10000)

# convert to a corpus. you need to identify the text field
tweets_corpus  <- corpus(tweets, text_field="text")

# see the output
summary(tweets_corpus,  n = 5)
class(tweets_corpus)
str(tweets_corpus)

# as you can see from this example, as soon as you convert your data to a basic dataframe, with the 
# text information stored in a column, it is super trivial to convert to a corpus object in quanteda. 

### 3.1.2 - from a json ----

# many ways to import jsons. Here I want to show you the `readtext` function from quanteda. 
# it actually allows you to importan many different types of data

pacman::p_load(readtext)

# save this as a json
tweets %>% jsonlite::write_json("tweets_json.json")

# load json
tweet_json <- readtext("tweets_json.json", text_field = "text")

# see class
tweet_json_corpus <- corpus(tweet_json)
class(tweet_json)
class(tweet_json_corpus)


### 3.1.3 - Quanteda Corpora ----
pacman::p_load(quanteda.corpora)

# Quanteda also offer several pre-loaded corpus. 
# From now on, we will work with the US State of the Union addresses
# See more corpus here: https://github.com/quanteda/quanteda.corpora

# see corpus
class(data_corpus_sotu)

# summary
summary(data_corpus_sotu)

## 3.2 - Working with a Corpus -----

# a corpus consists of: 
#   (1) documents: text + doc level data 
#   (2) corpus metadata 
#   (3) extras (settings)

# document-level variables (called docvars)
head(docvars(data_corpus_sotu)) 

class(docvars(data_corpus_sotu))

# corpus-level variables
meta(data_corpus_sotu)  

# ndoc identifies the number of documents in a corpus
ndoc(data_corpus_sotu)

# summary of the corpus (provides some summary statistics on the text combined with the metadata)
summary(data_corpus_sotu)  # note n default is 100

# SOTU number of tokens over time
ggplot(data = summary(data_corpus_sotu), 
                     aes(x = Date, 
                         y = Tokens, 
                         group = 1)) + 
  geom_line() + 
  geom_point() + 
  theme_bw()


# One nice feature of quanteda is that we can easily add metadata to the corpus object.
docvars(tweet_json_corpus, "text_length") <- nchar(tweet_json$text)
docvars(tweet_json_corpus)

# you can subset based on metadata
AZ.tweets <- corpus_subset(tweet_json_corpus, State=="AZ")
class(AZ.tweets)

# And then extract the text
AZ.texts <- quanteda::texts(tweet_json_corpus)


## 3.3 - Pre-Processing -------

# Pre-Processing is a key step to reduce complexity of text data. In `quanteda`, pre-processing happens at
# the tokens or at the document-feature matrix level, which means the corpus is preserved.
# Let's see some of the most commong pre-processing steps

### 3.3.1 Tokenization -----
tokens <- tokens(data_corpus_sotu)

# a nested list
str(tokens)

### 3.3.2 Remove Non-Informative Tokens -----


# tokens() is deliberately conservative, meaning that it does not remove anything 
# from the text unless told to do so.

# several other pre-processing steps can be done directly with the tokens function
?tokens

# Conservative
tokens(data_corpus_sotu[[241]])

# Remove punctuations
tokens(data_corpus_sotu[[241]],
       remove_punct = TRUE)

# Remove numbers
tokens(data_corpus_sotu[[241]],
       remove_punct = TRUE, 
       remove_numbers=TRUE)

# lower case
tokens(data_corpus_sotu[[241]],
       remove_punct = TRUE, 
       remove_numbers=TRUE) %>%
  tokens_tolower() 

### 3.3.3 Stopwords -----

tokens(data_corpus_sotu[[241]],
       remove_punct = TRUE, 
       remove_numbers=TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en"))

# what is this object?
stopwords("en") # just a list

# lets add some functional words
tokens(data_corpus_sotu[[241]],
       remove_punct = TRUE, 
       remove_numbers=TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(c(stopwords("en"), "madam", "mr", "speaker"))

### 3.2.3 Stemming --------------

tokens(data_corpus_sotu[[241]],
       remove_punct = TRUE, 
       remove_numbers=TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(c(stopwords("en"), "madam", "mr", "speaker")) %>%
  tokens_wordstem()


#Note that stemming is available in multiple languages:
  
tokens_wordstem(tokens("esto es un ejemplo"), language="es")
tokens_wordstem(tokens("isso é um exemplo"), language="pt")
tokens_wordstem(tokens("isso é um exemplo"), language="fr") # wrong stemming

### 3.3.4 n-grams -----

tokens(data_corpus_sotu[[241]],
       remove_punct = TRUE, 
       remove_numbers=TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(c(stopwords("en"), "madam", "mr", "speaker")) %>%
  tokens_ngrams()

## identifying meaningful n-grams (collocations)
pacman::p_load(quanteda.textstats)

# pre-process tokens
t_sotu <- tokens(data_corpus_sotu[[241]],
       remove_punct = TRUE, 
       remove_numbers=TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(c(stopwords("en")))

# identiy collocations
textstat_collocations(t_sotu) %>% arrange(-lambda) %>% slice(1:5)

tokens_ex = t_sotu %>%
       tokens_compound(list(c("carl", "marsha"), c("middle", "east"))) 

# see if it works
str_subset(unname(unlist(tokens_ex)), "middle")

## 3.4 - From text to Matrices - Converting tokens to DFMs -----

# a crucial part of any text as data task is to convert text to numerical representation. 
# as we saw in class, the workhorse model to represent text as numbers is by using the document-feature representation

# quanteda allows us to convert tokens to dfms quite easily

tokens_preproc = tokens(data_corpus_sotu,
       remove_punct = TRUE, 
       remove_numbers=TRUE, 
       remove_symbols=TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(c(stopwords("en"), "madam", "mr", "speaker")) %>%
  tokens_wordstem()


# convert to dfm
dfm_sotu <- tokens_preproc %>% 
              dfm()

# see main features
topfeatures(dfm_sotu)

# remove too frequent and rare words is also quite easy using dfm
dfm_sotu_trim <- dfm_sotu %>% 
          dfm_trim(min_docfreq = 0.05,
                   max_docfreq = 0.95,
                   docfreq_type ="prop", 
                   verbose = TRUE)

topfeatures(dfm_sotu_trim)

# see all features
featnames(dfm_sotu_trim)

#`dfm` has many useful options, including pre-processing

twdfm <- dfm(data_corpus_sotu, tolower=TRUE, stem=TRUE,
             remove_punct = TRUE, ngrams=1:3,
             verbose=TRUE, remove=c(
               stopwords("english")))

# throws a bunch of warnings
topfeatures(twdfm)

# wordcloud
pacman::p_load(quanteda.textplots)

textplot_wordcloud(dfm_sotu_trim,  random_order = FALSE,
                   rotation = .25, max_words = 100,
                   min_size = 0.5, max_size = 2.8,
                   color = RColorBrewer::brewer.pal(8, "Set3"))


# 4. Handling other languages  --------------

# Most latin languages are easy to tokenize, since words can be segmented by 
# spaces (i.e. word boundaries). This is not true for other languages like Chinese. 

# Quanteda can handle MANY languages (all major languages in Unicode). 

# See here an Chinese example taken from: https://quanteda.io/articles/pkgdown/examples/chinese.html

# And this quanteda tutorial for more examples: https://tutorials.quanteda.io/multilingual/


# 5. pretext from Denny and Spirling  --------------

#devtools::install_github("matthewjdenny/preText")

library(preText)

# Run at home (takes a few minutes to run)
# Example below taken from preText vignette: http://www.mjdenny.com/getting_started_with_preText.html
preprocessed_documents <- factorial_preprocessing(
  data_corpus_sotu[1:50],
  use_ngrams = FALSE,
  infrequent_term_threshold = 0.2,
  verbose = TRUE)


# see outputs
str(preprocessed_documents, max.level = 2)
head(preprocessed_documents$choices)


# get pretext scores
preText_results <- preText(preprocessed_documents,
                           dataset_name = "SOTU Speeches",
                           distance_method = "cosine",
                           num_comparisons = 20,
                           verbose = TRUE)


# score plots
preText_score_plot(preText_results)

# regression
regression_coefficient_plot(preText_results,
                            remove_intercept = TRUE)

# 6. Tidytext --------------------------

# The `tidytext` implements the tidy data approach to work with text in R. 
# If you are used to work with tidyverse, using the same principles for descriptive analysis using text will be easy task.

# in my experience, when I have to do tasks like the ones we saw this week, I usually prefer to use tidytext. 
# then, when it comes to fit more advanced models, I usually go to quanteda. 

# for this reason, I will provide here a quick example of using a tidytext. 
# If you want to learn more, see these boook and pretty much anything coming from Julia Silge:

## https://www.tidytextmining.com/

## https://smltar.com/

# 6.1 - The tidy text format --------------

## tidy data has a specific structure:
  
# a) Each variable is a column
# b) Each observation is a row
# c) Each type of observational unit is a table


## when working with, a tidy text format if defined as being a table with one-token-per-row. 

## This decision explodes the numbers of rows on your dataset, and it is not super computationally efficient. 
## however, it allows you to use all the tidyverse wrangling tools (group by, summarize, mutate, count, joins, etc..) easily in text. 


# tidying your text data: unnest_tokens()
tidy_tw <- tweets_s %>%
  mutate(id_tweets=1:nrow(.)) %>%
  unnest_tokens(words, text) #(output, input)

# see dimensions
dim(tweets_s)
dim(tidy_tw) # much longer

# tidying in sentence
tweets_s %>%
  mutate(id_tweets=1:nrow(.)) %>%
  unnest_tokens(sentences, text, token="sentences") %>%
  slice(1:100) %>%
  pull(sentences)

# in n-grams
tweets_s %>%
  mutate(id_tweets=1:nrow(.)) %>%
  unnest_tokens(sentences, text, token="ngrams", n=2) %>%
  slice(1:100) %>%
  pull(sentences)


# you can also move from dfm/corpus to a tidy dataframe 
data_corpus_sotu %>%
       tidy()

# and even cast from tidy to dfm: `cast_dfm()`
tweets_s %>%
  mutate(id_tweets=1:nrow(.)) %>%
  unnest_tokens(words, text)  %>%
  count(id_tweets, words)  %>%
  cast_dfm(id_tweets, words, n)

## 6.2 - Pre-Processing Steps ----
pacman::p_load(SnowballC)

# with a dataset in the tidyformat, all pre-processing steps are a mix of join operation and 
# vectorized data manipulation functions. Let's see examples

# see a chain with all the steps we learned before
tweet_pre = tweets_s %>%
  # already converts to lower and remove punctuation
  unnest_tokens(word, text) %>%
  # stop words
  anti_join(stop_words) %>%
  # any additional functional words
  anti_join(tibble(word=c("mr", "representative", "congressman"))) %>%
  # filter out digits
  filter(!str_detect(word, "[:digit:]")) %>%
  # stem
  mutate(word = wordStem(word, language = "en"))


## 6.3 - Descriptive analysis -----

# with data in that format, there is nothing new you need to learn to implement, for example, the 
# methods we say in Ban's article about measuring relative power. 

# see below an simple analyzes of words more associated with republicans and democrats on twitter

library(scales)

# frequency words
frequency_words <- tweet_pre %>%
                  select(Party, word) %>%
                  group_by(Party, word) %>%
                  summarize(total_words_per_party=n()) %>%
                  ungroup() %>%
                  filter(Party!="I")

# get total vocab size
all_words <- tweet_pre %>%
            count(Party) %>%
            filter(Party!="I")

# Merge
results <- left_join(frequency_words, all_words) %>%
             mutate(prop=total_words_per_party/n) %>%
             #untidy
            select(word, Party, prop) %>%
            pivot_wider(names_from=Party,
                        values_from=prop) %>%
            drop_na() %>%
            mutate(more=ifelse(D>R, "More Democrat", "More Republican"))

# Graph  
ggplot(results, aes(x = D, y = R, 
                     alpha = abs(D - R), 
                     color=more)) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5, alpha=.8) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  scale_color_manual(values=c("#5BBCD6","#FF0000"), name="") +
  theme(legend.position="none") +
  labs(y = "Proportion of Words (Republicans)", x = "Proportion of Words (Democrats)") +
  theme_minimal()



