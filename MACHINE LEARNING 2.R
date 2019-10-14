library(readr)
library(caret)
set.seed(123)
universal <- read_csv("UniversalBank.csv")
View(UniversalBank)
head(UniversalBank)
summary(UniversalBank)
library(dplyr)

#removing id and zipcode from data set
ub <- universal[,c(-1,-5)] 
str(ub)
table(ub$`Personal Loan`)# total numeber of loan acceptance and deny
prop.table(table(ub$`Personal Loan`))*100 # probality of loan acceptance and deny

# Assigning the labels to predictor
ub$`Personal Loan`<-factor(ub$`Personal Loan`,levels=c(0,1),labels = c("Deny","Accept"))
summary(ub)

# Creating a data parition with training data 60% and test data 40%
train_index<-createDataPartition(ub$Age,p=0.6,list=FALSE)
train_data<-ub[train_index,]
val_data<-ub[-train_index,]
test_index<-createDataPartition(ub$Age,p=0.2,list=FALSE)
test_data<-ub[test_index,]
traval_data<-ub[-test_index,]

summary(train_data)
summary(val_data)
summary(test_data)

#normalizing the data set
train.norm.df<-train_data[,-8]
val.norm.df<-val_data[,-8]
test.norm.df<-test_data[,-8]
traval.norm.df<-traval_data[,-8]

norm.values<-preProcess(train.norm.df,method = c("center","scale"))
train.norm.df<-predict(norm.values,train.norm.df)
val.norm.df<-predict(norm.values,val.norm.df)
test.norm.df<-predict(norm.values,test.norm.df)
traval.norm.df<-predict(norm.values,traval.norm.df)

#Applying knn with k value =1
library(FNN)
nn<- knn(train.norm.df,test=test.norm.df,cl=train_data$`Personal Loan`,k=1,prob = TRUE)

library(gmodels)
CrossTable(x=test_data$`Personal Loan`,y=nn,prop.chisq = FALSE)


accuracy.df <- data.frame(k = seq(1, 55, 1), accuracy = rep(0, 55))
for(i in 1:55) {
  knn.pred <- knn(train.norm.df, val.norm.df, 
                  cl = train_data$`Personal Loan`, k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, val_data$`Personal Loan`)$overall[1] 
}
accuracy.df

accuracy.df[which.max(accuracy.df$accuracy),]


# choice of k that balances between overfitting and ignoring the predictor information

# from the tunning K=4 is te best value after k=4, the model is said to be underfit for the model

# considering the customer scenario
a <- data.frame("Age" = 40, "Experience" = 10, "Income" = 84, "Family" = 2, "CCAvg" = 2, "Education"= 1, "Mortgage" = 0, "Personal Loan"= "Deny","Securities Account" = 0, "CD Account" = 0, "Online" = 1,"Credit Card" = 1)
b <- data.frame("Age" = 40, "Experience" = 10, "Income" = 84, "Family" = 2, "CCAvg" = 2, "Education"= 2, "Mortgage" = 0, "Personal Loan"= "Accept","Securities Account" = 0, "CD Account" = 0, "Online" = 1,"Credit Card" = 1)
c <- data.frame("Age" = 40, "Experience" = 10, "Income" = 84, "Family" = 2, "CCAvg" = 2, "Education"= 3, "Mortgage" = 0, "Personal Loan"= "Deny","Securities Account" = 0, "CD Account" = 0, "Online" = 1,"Credit Card" = 1)
test_pre<-as.data.frame(rbind(a,b,c))

test_pre.norm<-test_pre[,-8]
norm.values<- preProcess(test_pre.norm,method = c("center","scale"))
test_pre.norm<-predict(norm.values,test_pre.norm)
nn3<- knn(train.norm.df,test=test_pre.norm,cl=train_data$`Personal Loan`,k=1,prob = TRUE)
CrossTable(x=test_pre$`Personal.Loan`,y=nn3,prop.chisq = FALSE)
# The customer the loan will be declined.


# confusion matrix for the validation data that results from using the best k.

nn1<- knn(train.norm.df,test=val.norm.df,cl=train_data$`Personal Loan`,k=4,prob = TRUE)

confusionMatrix(knn.pred, val_data$`Personal Loan`)$overall[1] 
CrossTable(x=val_data$`Personal Loan`, y=nn1, prop.chisq = FALSE)

# Consider the customer scenario for the k =4 (best value)
d <- data.frame("Age" = 40, "Experience" = 10, "Income" = 84, "Family" = 2, "CCAvg" = 2, "Education"= 1, "Mortgage" = 0, "Personal Loan"= "Deny","Securities Account" = 0, "CD Account" = 0, "Online" = 1,"Credit Card" = 1)
e <- data.frame("Age" = 40, "Experience" = 10, "Income" = 84, "Family" = 2, "CCAvg" = 2, "Education"= 2, "Mortgage" = 0, "Personal Loan"= "Accept","Securities Account" = 0, "CD Account" = 0, "Online" = 1,"Credit Card" = 1)
f <- data.frame("Age" = 40, "Experience" = 10, "Income" = 84, "Family" = 2, "CCAvg" = 2, "Education"= 3, "Mortgage" = 0, "Personal Loan"= "Deny","Securities Account" = 0, "CD Account" = 0, "Online" = 1,"Credit Card" = 1)
test_pre<-as.data.frame(rbind(a,b,c))

test_pre.norm<-test_pre[,-8]
norm.values<- preProcess(test_pre.norm,method = c("center","scale"))
test_pre.norm<-predict(norm.values,test_pre.norm)
nn3<- knn(train.norm.df,test=test_pre.norm,cl=train_data$`Personal Loan`,k=4,prob = TRUE)
CrossTable(x=test_pre$`Personal.Loan`,y=nn3,prop.chisq = FALSE)


#Repartition the data, this time into training, validation, and test sets (50% : 30% : 20%).
ubank<-UniversalBank[,c(-1,-5)] 
View(ubank)
set.seed(15)
# Creating data partition
train_index1<-createDataPartition(ubank$Age,p=0.5,list=FALSE)
train_data1<-ubank[train_index1,] 
val_data1<-ubank[-train_index1,]  
test_index1<-createDataPartition(val_data1$Age,p=0.2,list=FALSE) 
test_data1<-val_data1[test_index1,] #20% testing data
val_data1<-val_data1[-test_index1,] 


train.norm.df1<-train_data1[,-8] #excluding perdictive indicator and assigning to the varialbe for normalizing the data
val.norm.df1<-val_data1[,-8]
test.norm.df1<-test_data1[,-8]


#Normalizing the data
norm.values1<-preProcess(train.norm.df1,method = c("center","scale"))
train.norm.df1<-predict(norm.values1,train.norm.df1)
val.norm.df1<-predict(norm.values1,val.norm.df1)
test.norm.df1<-predict(norm.values1,test.norm.df1)


# Assigning labels to the levels 
train_data1$`Personal Loan`<-factor(train_data1$`Personal Loan`,levels = c(0,1),labels = c("Deny","Accept"))
val_data1$`Personal Loan`<-factor(val_data1$`Personal Loan`,levels = c(0,1),labels = c("Deny","Accept"))
test_data1$`Personal Loan`<-factor(test_data1$`Personal Loan`,levels = c(0,1),labels = c("Deny","Accept"))


#Modelling KNN
library(FNN)
n_n<- knn(train.norm.df1,test=test.norm.df1,cl=train_data1$`Personal Loan`,k=4,prob = TRUE)
confusionMatrix(n_n,test_data1$`Personal Loan`)$overall[1]

###Hypertuning using validation####
accuracy.df1 <- data.frame(k = seq(1, 55, 1), accuracy = rep(0, 55))
for(i in 1:55) {
  knn.pred1 <- knn(train.norm.df1, val.norm.df1, 
                   cl = train_data1$`Personal Loan`, k = i)
  accuracy.df1[i, 2] <- confusionMatrix(knn.pred1, val_data1$`Personal Loan`)$overall[1] 
}
accuracy.df1

accuracy.df1[which.max(accuracy.df$accuracy),]

#choice of k that balances between overfitting is k=4;

library(gmodels)
CrossTable(x=test_data1$`Personal Loan`,y=n_n,prop.chisq = FALSE) 

# Accuracy with training (60%) and validation (40%) sets is 96.01%  and accuracy with training, validation, and test sets (50% : 30% : 20%) is 95.22% because of selection bias 








