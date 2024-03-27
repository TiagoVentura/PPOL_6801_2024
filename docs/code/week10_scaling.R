##############################################################################
# File-Name: week10_scaling.r
# Date: March 22, 2024
# author: Tiago Ventura
# course: PPOL 6801 - text as data
# topics: scaling models
# Machine: MacOS High Sierra
##############################################################################


# 0 - Introduction ----------------------------------------------------------

## Today we will learn about how to use text as data models to scale text in a ideological dimension. 
## We will see two core methods: 

# Wordscores for supervised methods

# Wordfish for unsupervised methods

## Some other tutorials available:
## quanteda wordscores: https://tutorials.quanteda.io/machine-learning/wordscores/
## quanteda wordfish: https://tutorials.quanteda.io/machine-learning/wordfish/

## Setup

#devtools::install_github("conjugateprior/austin")
library(austin) # just the wordscores
library(tidyverse) # for wrangling data
library(tidytext) # for 'tidy' manipulation of text data
library(quanteda) # tokenization power house
library(quanteda.textmodels)
library(wesanderson) # to prettify



# 1 - Wordscores ----------------------------------------------------------

# Laver et al. (2003) propose a supervised scaling technique called wordscores. 
# Let's replicate Table 1 from Laver and Benoit (2003) using the austin package.


# Let's first see an example using Austin Package:

# call data
data(lbg)

# same table as in the paper
head(lbg)


# let's get reference docs
ref <- getdocs(lbg, 1:5)
ref

# give scores to the reference documents
A_score <- c(-1.5,-0.75,0,0.75,1.5)

# what is this function?
?classic.wordscores

# run the model
ws <- classic.wordscores(ref, scores=A_score)

# see scores
ws$pi

# Estimating on a virgin text
vir <- getdocs(lbg, 'V1')

# predic
predict(ws, newdata=vir)


## 1.2 - Example with Quanteda Wordscore ----------------------------------
library(quanteda)
library(quanteda.textmodels)
library(quanteda.textplots)
library(quanteda.corpora)

# get the corpus. Germans manifestos
corp_ger <- download(url = "https://www.dropbox.com/s/uysdoep4unfz3zp/data_corpus_germanifestos.rds?dl=1")

# see it
summary(corp_ger)

# tokenize texts
toks_ger <- tokens(corp_ger, remove_punct = TRUE)

# create a document-feature matrix
dfmat_ger <- dfm(toks_ger) %>% 
  dfm_remove(pattern = stopwords("de"))

# notice scores are here already for the training data
corp_ger$ref_score


# apply Wordscores algorithm to document-feature matrix
tmod_ws <- textmodel_wordscores(dfmat_ger, y = corp_ger$ref_score, smooth = 1)


# see some words
str(tmod_ws,2)
summary(tmod_ws)

# capture words as a tibble
wordscores_df <- tmod_ws$wordscores %>% 
  as_tibble() %>% 
  mutate(words=names(tmod_ws$wordscores))

tmod_ws$scale

# Make a graph about the words more heavily associated with the left and with the right

# right words
words_r  <- wordscores_df %>%
  arrange(desc(value)) %>%
  slice(1:10)


# left_words
words_l  <- wordscores_df %>%
  arrange(value) %>%
  slice(1:10)

# bind and rescale
# Rescale scores to be between -1 and 1
min_score <- min(wordscores_df$value)
max_score <- max(wordscores_df$value)
wordscores_resscaled = bind_rows(words_r, words_l) %>%
  arrange(value) %>%
  mutate(rescaled_scores = -1 + (value - min_score) * (2 / (max_score - min_score)), 
         fill_=ifelse(rescaled_scores > 0, "Right", "Left"), 
         words=fct_rev(fct_inorder(words)))


# plot for the words
ggplot(wordscores_resscaled, 
       aes(x=words, y=rescaled_scores, label = words, fill = fill_)) +
  geom_col(show.legend = FALSE, alpha=.8) +
  geom_text(size = 4) +
  coord_flip() +
  scale_y_continuous() +
  theme_minimal(base_size = 12) +
  theme(axis.text.y  = element_blank()) +
  scale_fill_manual(values = wes_palette("BottleRocket2"))


# 2 - Wordfish ----------------------------------------------------------

# Slapin and Proksch (2008) propose an unsupervised scaling model that places texts in a one-dimensional scale. 

# The model basically consist on learning words that discriminate better certain documents. 

# data: we will work with inaugural speeches from US presidents
# download here: https://www.dropbox.com/scl/fi/hgwwexq7oe6kux4h54opi/inaugTexts.xlsx?rlkey=5x3evugu46rs7pfbga3v3z2f8&dl=0
library(here)
us_pres <- readxl::read_xlsx(path = here("data", "inaugTexts.xlsx"))
head(us_pres)

# The text is pretty clean, so we can change it into a corpus object and then a dfm and apply textmodel_wordfish():
?textmodel_wordfish

corpus_us_pres <- corpus(us_pres,
                         text_field = "inaugSpeech",
                         unique_docnames = TRUE)

summary(corpus_us_pres)

# tokenization
toks_us_pres <- tokens(corpus_us_pres,
                       remove_numbers = TRUE, # Thinks about this
                       remove_punct = TRUE, # Remove punctuation!
                       remove_url = TRUE) # Might be helpful

toks_us_pres <- tokens_remove(toks_us_pres,
                              # Should we though? See Denny and Spirling (2018)
                              c(stopwords(language = "en")),
                              padding = F)

toks_us_pres <- tokens_wordstem(toks_us_pres, language = "en")

dfm_us_pres <- dfm(toks_us_pres)


# modelin
#Does not really matter what the starting values are, they just serve as anchors for the 
# relative position of the rest of the texts. In this case, I chose Kennedy and Nixon.  
wfish_us_pres <- textmodel_wordfish(dfm_us_pres, dir = c(28,30)) 
summary(wfish_us_pres)


# Get document schores
?quanteda.textmodels::predict.textmodel_wordfish
wfish_preds <- predict(wfish_us_pres, interval = "confidence")

# Tidy everything up:
posi_us_pres <- data.frame(docvars(corpus_us_pres),
                           wfish_preds$fit) %>%
  arrange(fit)

# Plot
posi_us_pres %>%
  ggplot(aes(x = fit, y = reorder(President,fit), xmin = lwr, xmax = upr, color = party)) +
  geom_point(alpha = 0.8) +
  geom_errorbarh(height = 0) +
  labs(x = "Position", y = "", color = "Party") +
  scale_color_manual(values = wes_palette("BottleRocket2")) +
  theme_minimal() +
  ggtitle("Estimated Positions")

# bit of weird results, specially with nixon been so close to clinton

# Time seems to be the main thing here:
posi_us_pres %>%
  ggplot(aes(y = -fit, x = Year, ymin = -lwr, ymax = -upr, color = party)) +
  geom_point(alpha = 0.8) +
  geom_errorbar() +
  labs(x = "Year", y = "Position", color = "Party") +
  scale_color_manual(values = wes_palette("BottleRocket2")) +
  theme_minimal() +
  ggtitle("Estimated Positions")


# let's try to be less agressive with the pre-processing
# Tokenization only removing punctuation
toks_us_pres2 <- tokens(corpus_us_pres,
                        remove_punct = TRUE) 

dfm_us_pres2 <- dfm(toks_us_pres2)
wfish_us_pres <- textmodel_wordfish(dfm_us_pres2, dir = c(28,30))  

# Get predictions:
wfish_preds <- predict(wfish_us_pres, interval = "confidence")

# Tidy everything up:
posi_us_pres <- data.frame(docvars(corpus_us_pres),
                           wfish_preds$fit) %>%
  arrange(fit)

# Plot
posi_us_pres %>%
  ggplot(aes(x = -fit, y = reorder(President,fit), xmin = -lwr, xmax = -upr, color = party)) +
  geom_point(alpha = 0.8) +
  geom_errorbarh(height = 0) +
  labs(x = "Position", y = "", color = "Party") +
  scale_color_manual(values = wes_palette("BottleRocket2")) +
  theme_minimal() +
  ggtitle("Estimated Positions (No Pre-Processing")

# we so slightly better

# probably time here is explaining better the differences in the words. 
# So ideally I would try to actually be more agressive with pre-processing and take words that matter more
# for what I am trying to estimate. 