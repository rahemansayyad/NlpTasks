setwd("C:/Users/sraheman/Documents/r")

############ Data Gathering #################
library(vosonSML)
library(magrittr)

# Google api key to fetch comments from youtube
apikey <- "xxxxxxxxxxxxxxxxxxxxxxx"

# authentication call
key <- Authenticate("youtube",apikey)

# video id for which we want to pull the comments from
video <- c('Q4VGQPk2Dl8')

# fetching comments from youtube
ytdata <- key %>% Collect(video)

# analyzing collected data
str(ytdata)

# Writing to a file
write.csv(ytdata, file='teslaYoutubeComments.csv', row.names = F)

# reading raw data
data <- read.csv('teslaYoutubeComments.csv', header = T)

# Extracting only the interested column from data
comments <- data$Comment

# slicing the review to reduce the dataset size
comments <- comments[1:2000]

############ Various libararies used ################
library(devtools)
library(ggplot2)
library(lubridate)
library(scales)
library(tm)
library(stringr)
library(reshape2)
library(dplyr)
library(plyr)
library(readxl)
library(sentimentr)
library(wordcloud)
library(textreg)
library(RTextTools)
library(e1071)
library(caret)

############## Data Preprocessing #############
#creating a vector of reviews
wordCorpus <- VCorpus(VectorSource(comments))

#looking at frequency of word count
dtm <- DocumentTermMatrix(wordCorpus)
fullfreq <- findFreqTerms(dtm, 5)

clean_corpus <- function(corpus)
{
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"),"elon","musk","get","think","make","electric","now","india","just","will","model","one","cars","2020","apple","audience","car","chart","crowd","launch","tesla","etc","can","people","years","world","singapore"))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, PlainTextDocument) 
  return(corpus)
  
}

cleaned_word_corp <- clean_corpus(wordCorpus)

#clean_word_corp [[20]] [1] #cleaned comment
#comments[20]   #original comment

# Build a term-document matrix
dtm <- TermDocumentMatrix(cleaned_word_corp)
dtm_m <- as.matrix(dtm)

# Sort by descreasing value of frequency
dtm_v <- sort(rowSums(dtm_m),decreasing=TRUE)
dtm_d <- data.frame(word = names(dtm_v),freq=dtm_v)
# Display the top 5 most frequent words
head(dtm_d, 5)

# Plot the most frequent words
barplot(dtm_d[1:5,]$freq, las = 2, names.arg = dtm_d[1:5,]$word,
        col ="lightgreen", main ="Top 5 most frequent words",
        ylab = "Word frequencies")

#generate word cloud
set.seed(1234)
wordcloud(words = dtm_d$word, freq = dtm_d$freq, min.freq = 5,
          max.words=100, random.order=FALSE, rot.per=0.40, 
          colors=brewer.pal(8, "Dark2"))

# converting to character vector
char_vc <- convert.tm.to.character(clean_word_corp)

# Using lexion to get sentiment for individual review
google_lexicon <- lexicon::hash_sentiment_socal_google
sent_df <- char_vc %>% get_sentences() %>% sentiment_by(polarity_dt = google_lexicon)

# creating a dataframe to get reviews and there avg sentiment
dfr <- data.frame(comments = char_vc, sentiment = sent_df$ave_sentiment)

# removing neutral reviews
dfr <- dfr[!dfr$sentiment == 0,]

# labelling reviews as positive and negative
dfr$sentiment <- ifelse(dfr$sentiment > 0, 1, 0)

# look at DTM
dtm <- DocumentTermMatrix(clean_word_corp)
fullfreq <- findFreqTerms(dtm, 5)

# 75% of the sample size
smp_size <- floor(0.75 * nrow(dfr))

# set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(dfr)), size = smp_size)
df.train <- dfr[train_ind, ]
df.test <- dfr[-train_ind, ]

# training the Naive bayes classifier
classifier_NB <- naiveBayes(df.train$comments, as.factor(df.train$sentiment) , laplace = 0.5)

# test the predictions
pred_NB <- predict(classifier_NB, newdata = df.test$comments)

table("Predictions" = pred_NB,  "Actual" = df.test$sentiment)

# creating confusin matrix
conf.mat_NB <- confusionMatrix(pred_NB, as.factor(df.test$sentiment))

# look at accuracy
conf.mat_NB$overall['Accuracy']
