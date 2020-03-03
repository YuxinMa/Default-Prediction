

# Load data
traindata <- read.csv('/Users/yuxinma/Downloads/dataverse_files/traindata.csv')
traindata <- traindata[,-1] # Delete first column
colnames(traindata) <- c('y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10') # Change column name



# Check NAs and fill with KNN algorithm
library(mice)
library(VIM)
aggr(traindata,prop=FALSE,numbers=TRUE) # Check number of NAs 
md.pattern(traindata) # Specific numbers
library(DMwR)
traindata=knnImputation(traindata,k=10,meth='weighAvg')


# Dealing with outliers
# Credit cards applicants must between 18-65; Check and proceed univariate outliers detection
traindata=traindata[traindata$x2<=65&traindata$x2>=18,]
# x2 clear
# Multivariates outliers detection
boxplot(traindata1[,c('x3','x7','x9')])
traindata1=traindata1[-which(traindata1$x3 == 96|traindata1$x3 == 98),] # Delete outliers
# Check correlation between variables。
traindata_cor=cor(traindata1[,1:11])
library(corrplot)
corrplot(traindata_cor, method='number')


# SMOTE
# Apply SMOTE to the standardized dataset
data=read.csv('/Users/yuxinma/Downloads/dataverse_files/standardized_data.csv',sep=',')
table(data$y1)
## now using SMOTE to create a more "balanced problem"
## parameters:
## perc.over oversampling: 100/100, 1 time，minority generates 1*4824=4824 samples，
## plus orginal 4824 samples, total 9648 samples
## perc.under undersampling: 1400/100, 14 times, majority selects 4824*14=67536 samples
library(DMwR)
attach(data)
data$y1=factor(ifelse(data$y1 == "1","1","0")) 
newData=SMOTE(y1~.,data, perc.over = 900,perc.under = 150)
table(newData$y1)
write.csv(newData,'/Users/yuxinma/Downloads/dataverse_files/smote_standardized_data.csv',row.names = FALSE)
## Visually check the created data
par(mfrow = c(1, 2))
mycolor <- rainbow(3, alpha=0.02)
plot(data$x1, data$x2,pch=19+as.integer(data$y1),col=mycolor[as.integer(data$y1)+2],main = "Original Data")
plot(newData$x1, newData$x2,pch=19+as.integer(data$y1),col=mycolor[as.integer(data$y1)+2],
     main = "SMOTE'd Data")
table(newData$y1)


