#' Customer Propensity Scoring Pipeline
#'
#' Author: Yves Agbre
#' Course: CSCI E-96 - Data Mining for Business
#' Institution: Harvard University, Division of Continuing Education
#' Professor: Ted Kwartler
#' Date: December 2021
#'
#' This script builds and compares 4 classification models to predict
#' which prospective bank customers are most likely to accept a marketing offer.
#' The winning model scores prospective customers and outputs the top 100 targets.

# ==============================================================================
# SETUP
# ==============================================================================
options(scipen = 999)

library(vtreat)         # Variable treatment for categorical encoding
library(MLmetrics)      # Machine learning metrics (Accuracy, ConfusionMatrix)
library(pROC)           # ROC curve analysis
library(ggplot2)        # Visualization
library(ggthemes)       # Plot themes
library(dplyr)          # Data manipulation
library(caret)          # Model training framework (train, confusionMatrix)
library(rpart.plot)     # Decision tree visualization
library(randomForest)   # Random Forest implementation
library(e1071)          # SVM / Naive Bayes (caret dependency)
library(plyr)           # Data manipulation utilities
library(class)          # KNN implementation

# ==============================================================================
# DATA LOADING & JOINING
# ==============================================================================
# Four data sources joined on household ID to create a unified customer profile:
#   - Customer marketing results (target variable: accepted offer Y/N)
#   - Household vehicle data
#   - Household credit data
#   - Household Axiom demographic data

currCustMktgData <- read.csv('data/CurrentCustomerMktgResults.csv')
hhVehData        <- read.csv('data/householdVehicleData.csv')
hhCcData         <- read.csv('data/householdCreditData.csv')
hhAxData         <- read.csv('data/householdAxiomData.csv')

# Multi-source join on HHuniqueID
allCustData <- currCustMktgData %>%
  left_join(hhVehData, by = 'HHuniqueID') %>%
  left_join(hhCcData,  by = 'HHuniqueID') %>%
  left_join(hhAxData,  by = 'HHuniqueID')

# Classification target - ensure R treats as factor, not integer
allCustData$Y_AcceptedOffer <- as.factor(allCustData$Y_AcceptedOffer)
allCustData <- allCustData %>% relocate(Y_AcceptedOffer, .after = last_col())

# ==============================================================================
# TRAIN/TEST SPLIT
# ==============================================================================
set.seed(1234)
idx       <- sample(1:nrow(allCustData), nrow(allCustData) * 0.8)
trainData <- allCustData[idx, ]
testData  <- allCustData[-idx, ]

cat("Training set:", nrow(trainData), "rows\n")
cat("Test set:    ", nrow(testData), "rows\n")

# ==============================================================================
# FEATURE ENGINEERING (VTREAT)
# ==============================================================================
# Vtreat handles categorical variable encoding, missing values, and
# high-cardinality features automatically

xVars <- c('Communication', 'LastContactMonth', 'NoOfContacts',
           'carModel', 'HHInsurance', 'CarLoan',
           'Job', 'Marital', 'Education', 'past_Outcome')
yVars <- 'Y_AcceptedOffer'

plan <- designTreatmentsC(trainData, xVars, yVars, 'Accepted')

treatedTrain <- prepare(plan, trainData)
treatedTest  <- prepare(plan, testData)

# ==============================================================================
# MODEL 1: RANDOM FOREST
# ==============================================================================
rfFit <- train(Y_AcceptedOffer ~ .,
               data = treatedTrain,
               method = "rf",
               verbose = FALSE,
               ntree = 50,
               tuneGrid = data.frame(mtry = 1))

plot(varImp(rfFit), top = 20, main = "Random Forest - Variable Importance")

# Evaluate
rfTrainPreds <- predict(rfFit, treatedTrain)
rfTestPreds  <- predict(rfFit, treatedTest)

caret::confusionMatrix(rfTrainPreds, as.factor(treatedTrain$Y_AcceptedOffer))
caret::confusionMatrix(rfTestPreds,  as.factor(treatedTest$Y_AcceptedOffer))

rfTrainAccuracy <- Accuracy(rfTrainPreds, treatedTrain$Y_AcceptedOffer)
rfTestAccuracy  <- Accuracy(rfTestPreds,  treatedTest$Y_AcceptedOffer)

# ==============================================================================
# MODEL 2: DECISION TREE (RPART)
# ==============================================================================
dtFit <- train(Y_AcceptedOffer ~ .,
               data = treatedTrain,
               method = "rpart",
               tuneGrid = data.frame(cp = c(0.1, 0.01, 0.05, 0.07)),
               control = rpart.control(minsplit = 1, minbucket = 2))

plot(dtFit, main = "Decision Tree - Complexity Parameter vs Accuracy")
prp(dtFit$finalModel, extra = 1)

# Evaluate
dtTrainPreds <- predict(dtFit, treatedTrain)
dtTestPreds  <- predict(dtFit, treatedTest)

caret::confusionMatrix(dtTrainPreds, as.factor(treatedTrain$Y_AcceptedOffer))
caret::confusionMatrix(dtTestPreds,  as.factor(treatedTest$Y_AcceptedOffer))

rpartTrainAccuracy <- Accuracy(dtTrainPreds, treatedTrain$Y_AcceptedOffer)
rpartTestAccuracy  <- Accuracy(dtTestPreds,  treatedTest$Y_AcceptedOffer)

# ==============================================================================
# MODEL 3: LOGISTIC REGRESSION (with backward stepwise selection)
# ==============================================================================
# Convert target to binary for GLM
treatedTrainGLM <- treatedTrain
treatedTestGLM  <- treatedTest
treatedTrainGLM$Y_AcceptedOffer <- ifelse(treatedTrainGLM$Y_AcceptedOffer == 'Accepted', 1, 0)
treatedTestGLM$Y_AcceptedOffer  <- ifelse(treatedTestGLM$Y_AcceptedOffer == 'Accepted', 1, 0)

glmFit <- glm(Y_AcceptedOffer ~ .,
              data = treatedTrainGLM,
              family = 'binomial')

# Backward variable selection to reduce multicollinearity
glmBestFit <- step(glmFit, direction = 'backward')
summary(glmBestFit)

cat("Full model coefficients: ", length(coefficients(glmFit)), "\n")
cat("Reduced model coefficients:", length(coefficients(glmBestFit)), "\n")

# Evaluate
trainPreds <- predict(glmBestFit, treatedTrainGLM, type = 'response')
testPreds  <- predict(glmBestFit, treatedTestGLM,  type = 'response')

cutoff <- 0.5
trainY <- ifelse(trainPreds >= cutoff, 1, 0)
testY  <- ifelse(testPreds >= cutoff, 1, 0)

allCustDataGLM <- allCustData
allCustDataGLM$Y_AcceptedOffer <- ifelse(allCustDataGLM$Y_AcceptedOffer == 'Accepted', 1, 0)

glmTrainAccuracy <- Accuracy(trainY, allCustDataGLM$Y_AcceptedOffer[idx])
glmTestAccuracy  <- Accuracy(testY,  allCustDataGLM$Y_AcceptedOffer[-idx])

# ==============================================================================
# MODEL 4: K-NEAREST NEIGHBORS
# ==============================================================================
knnFit <- train(Y_AcceptedOffer ~ .,
                data = treatedTrain,
                method = "knn",
                preProcess = c("center", "scale"),
                tuneLength = 47)

plot(knnFit, main = "KNN - Accuracy vs K")

# Evaluate
knnTrainPreds <- predict(knnFit, treatedTrain)
knnTestPreds  <- predict(knnFit, treatedTest)

caret::confusionMatrix(knnTrainPreds, as.factor(treatedTrain$Y_AcceptedOffer))
caret::confusionMatrix(knnTestPreds,  as.factor(treatedTest$Y_AcceptedOffer))

knnTrainAccuracy <- Accuracy(knnTrainPreds, treatedTrain$Y_AcceptedOffer)
knnTestAccuracy  <- Accuracy(knnTestPreds,  treatedTest$Y_AcceptedOffer)

# ==============================================================================
# MODEL COMPARISON
# ==============================================================================
accuracyTable <- data.frame(
  Model = c('Random Forest', 'Decision Tree', 'Logistic Regression', 'KNN'),
  Train = c(rfTrainAccuracy, rpartTrainAccuracy, glmTrainAccuracy, knnTrainAccuracy),
  Test  = c(rfTestAccuracy,  rpartTestAccuracy,  glmTestAccuracy,  knnTestAccuracy)
)
accuracyTable <- accuracyTable[order(accuracyTable$Test, decreasing = TRUE), ]
print(accuracyTable)

# ==============================================================================
# SCORING PROSPECTIVE CUSTOMERS
# ==============================================================================
# Apply the best model to score new prospects and identify top 100 targets

prospects <- read.csv('data/ProspectiveCustomers.csv')

# Join with external data (same sources as training)
prospectsJoined <- prospects %>%
  left_join(hhVehData, by = 'HHuniqueID') %>%
  left_join(hhCcData,  by = 'HHuniqueID') %>%
  left_join(hhAxData,  by = 'HHuniqueID')

prospectsJoined <- prospectsJoined %>% relocate(Y_AcceptedOffer, .after = last_col())

# Apply treatment plan and score with KNN
prospectsTreated  <- prepare(plan, prospectsJoined)
prospectProbKNN   <- predict(knnFit, prospectsTreated, type = 'prob')

# Join probabilities back to customer IDs
prospectsResults <- cbind(HHuniqueID = prospects$HHuniqueID, prospectProbKNN)
prospectsResults <- left_join(prospectsResults, prospectsJoined, by = 'HHuniqueID')

# Top 100 most likely to accept
top100 <- prospectsResults %>%
  arrange(desc(Accepted)) %>%
  head(100) %>%
  select(HHuniqueID, Accepted, past_Outcome, PrevAttempts, NoOfContacts,
         DaysPassed, Communication, carModel, carYr, Job, Marital,
         Education, Age, AffluencePurchases, headOfhouseholdGender,
         CarLoan, EstRace)

write.csv(top100, 'output/top_100_prospects.csv', row.names = FALSE)

# ==============================================================================
# PROSPECT ANALYSIS VISUALIZATIONS
# ==============================================================================
ggplot(top100, aes(x = Job, y = Accepted)) +
  geom_bar(stat = "identity") + coord_flip() +
  ggtitle("Top 100 Prospects - Acceptance Probability by Job")

ggplot(top100, aes(x = Education, y = Accepted)) +
  geom_bar(stat = "identity") + coord_flip() +
  ggtitle("Top 100 Prospects - Acceptance Probability by Education")

ggplot(top100, aes(x = Marital, y = Accepted)) +
  geom_bar(stat = "identity") + coord_flip() +
  ggtitle("Top 100 Prospects - Acceptance Probability by Marital Status")

ggplot(top100, aes(x = Communication, y = Accepted)) +
  geom_bar(stat = "identity") + coord_flip() +
  ggtitle("Top 100 Prospects - Acceptance Probability by Channel")
