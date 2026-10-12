## ----init---------------------------------------------------------------------
## Init by deleting all variables and functions
rm(list=ls())

## Set the working directory. Change this to the location of the example on the computer. Note that "/" is always used in R, also in Windows
setwd(".")

## Use the CTSM-R package, note that first the package must be installed, see the Installation section in the CTSM-R Userguide
library(ctsmr)

## List with global parameters
prm <- list()
## Number of threads used by CTSM-R for the estimation computations
prm$threads <- 1


## ----sourcefuntions, results="hide"-------------------------------------------
## Source the scripts with functions in the "functions" folder. Just a neat way of arranging helping functions in R
sapply(dir("functions", full.names=TRUE), source)


## ----readData-----------------------------------------------------------------
## Read the data into a data.frame
X <- read.csv("inputPRBS1.csv",sep=";",header=TRUE)
## X$t is now hours since start of the experiment. Create a column in the POSIXct format for plotting etc.
X$timedate <- asP("2009-02-05 14:26:00)") + X$t * 3600


## ----plotData, fig.height=6---------------------------------------------------
## Plot the time series (see "functions/setpar.R" to see the plot setup function)
setpar("ts", mfrow=c(4,1))
gridSeq <- seq(asP("2009-01-01"),by="days",len=365)
## 
plot(X$timedate,X$yTi,type="n",xlab="",ylab="yTi ($^{\\circ}$C)")
abline(v=gridSeq,h=0,col="grey85",lty=3)
lines(X$timedate,X$yTi)
## 
plot(X$timedate,X$Ta,type="n",xlab="",ylab="Ta ($^{\\circ}$C)")
abline(v=gridSeq,h=0,col="grey85",lty=3)
lines(X$timedate,X$Ta)
## 
plot(X$timedate,X$Ph,type="n",xlab="",ylab="Ph (kW)")
abline(v=gridSeq,h=0,col="grey85",lty=3)
lines(X$timedate,X$Ph)
##
plot(X$timedate,X$Ps,type="n",xlab="",ylab="Ps (kw/m$^2$)")
abline(v=gridSeq,h=0,col="grey85",lty=3)
lines(X$timedate,X$Ps)
axis.POSIXct(1, X$timedate, xaxt="s", format="%Y-%m-%d")


## ----TiTeNew,tidy=FALSE-------------------------------------------------------
## Generate a new object of class ctsm
model <- ctsm$new()
## Add system equations and thereby also states
model$addSystem(dTi ~ ( 1/(Ci*Rie)*(Te-Ti) + Aw/Ci*Ps + 1/Ci*Ph )*dt + exp(p11)*dw1)
model$addSystem(dTe ~ ( 1/(Ce*Rie)*(Ti-Te) + 1/(Ce*Rea)*(Ta-Te) )*dt + exp(p22)*dw2)
## Set the names of the inputs
model$addInput(Ta,Ps,Ph)


## ----addObs-------------------------------------------------------------------
## Set the observation equation: Ti is the state, yTi is the measured output
model$addObs(yTi ~ Ti)
## Set the variance of the measurement error
model$setVariance(yTi ~ exp(e11))


## ----initialValues,tidy=FALSE-------------------------------------------------
## Set the initial value (for the optimization) of the states at the start time point
model$setParameter(  Ti = c(init=15  ,lb=0     ,ub=25) )
model$setParameter(  Te = c(init=5   ,lb=-20   ,ub=25) )
## Set the initial value of the parameters for the optimization
model$setParameter(  Ci = c(init=1   ,lb=1E-5  ,ub=20) )
model$setParameter(  Ce = c(init=2   ,lb=1E-5  ,ub=20) )
model$setParameter( Rie = c(init=10  ,lb=1E-5  ,ub=50) )
model$setParameter( Rea = c(init=10  ,lb=1E-5  ,ub=50) )
model$setParameter(  Aw = c(init=20  ,lb=0.1   ,ub=200))
model$setParameter( p11 = c(init=1   ,lb=-50   ,ub=10) )
model$setParameter( p22 = c(init=1   ,lb=-50   ,ub=10) )
model$setParameter( e11 = c(init=-1  ,lb=-50   ,ub=10) )


## ----estimate,results="hide"--------------------------------------------------
## Run the parameter optimization
fit <- model$estimate(data=X, threads=prm$threads)


## ----summaryfit---------------------------------------------------------------
## See the summary of the estimation
print(summary(fit,extended=TRUE))


## ----oneStepPred--------------------------------------------------------------
## Calculate the one-step predictions of the state (i.e. the residuals)
tmp <- predict(fit)[[1]]
## Calculate the residuals and put them with the data in a data.frame X
X$residuals <- X$yTi - tmp$output$pred$yTi
X$yTiHat <- tmp$output$pred$yTi


## ----residualsACF, fig.height=3-----------------------------------------------
## Plot the auto-correlation function and cumulated periodogram in a new window
par(mfrow=c(1,3))
## The blue lines indicates the 95 confidence interval, meaning that if it is
## white noise, then approximately 1 out of 20 lag correlations will be slightly outside
acf(X$residuals, lag.max=6*12, main="Residuals ACF")
## The periodogram is the estimated energy spectrum in the signal
spec.pgram(X$residuals, main="Raw periodogram")
## The cumulated periodogram
cpgram(X$residuals, main="Cumulated periodogram")


## ----residualsplot------------------------------------------------------------
## Plot the time series (see "functions/setpar.R" to see the plot setup function)
setpar("ts", mfrow=c(5,1))
gridSeq <- seq(asP("2009-01-01"), by="days", len=365)
##
plot(X$timedate, X$residuals, xlab="yTi ($^{\\circ}$C)", ylab="", type="n")
abline(v=gridSeq, h=0, col="grey92")
lines(X$timedate, X$residuals)
##
plot(X$timedate, X$yTi, ylim=range(X[ ,c("yTi","yTiHat")]), type="n", xlab="", ylab="yTi, yTiHat ($^{\\circ}$C)")
abline(v=gridSeq, h=0, col="grey85", lty=3)
lines(X$timedate, X$yTi)
lines(X$timedate, X$yTiHat, col=2)
legend("bottomright", c("Measured","Predicted"), lty=1, col=1:2, bg="grey95")
##
plot(X$timedate, X$Ph, type="n", xlab="", ylab="Ph (kW)")
abline(v=gridSeq, h=0, col="grey85", lty=3)
lines(X$timedate, X$Ph)
##
plot(X$timedate, X$Ps, type="n", xlab="", ylab="Ps (kw/m$^2$)")
abline(v=gridSeq, h=0, col="grey85", lty=3)
lines(X$timedate, X$Ps)
##
plot(X$timedate, X$Ta, type="n", xlab="", ylab="Ta ($^{\\circ}$C)")
abline(v=gridSeq, h=0, col="grey85", lty=3)
lines(X$timedate, X$Ta)
axis.POSIXct(1, X$timedate, xaxt="s", format="%Y-%m-%d")


## ----executeTiTe,results="hide"-----------------------------------------------
fitTiTe <- TiTe(X)


## ----analyzeFit,results="hide"------------------------------------------------
analyzeFit(fitTiTe ,tPer=c("2009-02-07","2009-02-08"),plotACF=FALSE)


## ----TiTeThExecute,results="hide"---------------------------------------------
fitTiTeTh <- TiTeTh(X)


## ----analyzeTiTeThEstimates, size="scriptsize"--------------------------------
analyzeFit(fitTiTeTh,plotACF=FALSE,plotSeries=FALSE)


## ----analyzeTiTeThACF,fig.height=3,results="hide", cache=FALSE----------------
analyzeFit(fitTiTeTh,plotSeries=FALSE)


## ----analyzeTiTeThSeries,results="hide"---------------------------------------
analyzeFit(fitTiTeTh,plotACF=FALSE)


## ----estimate-hlc-value-------------------------------------------------------
## The estimated HLC-value 
i <- which(names(fitTiTeTh$xm)%in%c("Rea","Rie"))
HLC <- 1/sum(fitTiTeTh$xm[i])
HLC*1000 ## W/C
## The covariance for the two estimated R values
cov <- diag(fitTiTeTh$sd[i]) %*% fitTiTeTh$corr[i,i] %*% diag(fitTiTeTh$sd[i])


## ----jacobian-----------------------------------------------------------------
## The Jacobian, the derived of the HLC-value with respect to each estimate in fitTiTeTh$xm[i]
J <- t( sapply(1:length(i), function(ii,x){ -1/sum(x)^2 }, x=fitTiTeTh$xm[i]) )
## The estimated variance of U
varHLC <- J %*% cov %*% t(J)    
## and standard deviance
sdHLC <- sqrt(varHLC)
## Return the confidence interval
c(HLC-1.96*sdHLC, HLC+1.96*sdHLC)*1000


## ----multivar-----------------------------------------------------------------
## Needed for multivariate normal distribution simulation
require(MASS)
## Generate multivariate normal random values
Rsim <- mvrnorm(n=1000000, mu=fitTiTeTh$xm[i], Sigma=cov)
## For each realization calculate the HLC-value
HLCsim <- 1/apply(Rsim, 1, sum)
## Estimate the 2.5% and 97.5% quantiles of the simulated values as a confidence interval
quantile(HLCsim, probs=c(0.025,0.975))*1000


## ----likelihoodratiotest------------------------------------------------------
## Take the results of both models
small <- fitTiTe
large <- fitTiTeTh
## Calculate the logLikelihood for both models from their fit
logLikSmallModel <- small$loglik
logLikLargeModel <- large$loglik
## Calculate lambda
chisqStat <- -2 * (logLikSmallModel - logLikLargeModel)
## It this gives a p-value smaller than confidence limit, i.e. 5\%, then the
## larger model is significant better than the smaller model
prmDiff <- large$model$NPARAM - small$model$NPARAM
## The p-value of the test
1 - pchisq(chisqStat, prmDiff)


## ----savefit,results="hide"---------------------------------------------------
save(fitTiTeTh,file="fitTiTeTh.rda")


## ----loadfit------------------------------------------------------------------
load("fitTiTeTh.rda")


## ----check1, include=FALSE----------------------------------------------------
val <- summary(fitTiTeTh)
exp(val$coefficients["p11","Estimate"]) / sqrt(12)
exp(val$coefficients["p22","Estimate"]) / sqrt(12)
exp(val$coefficients["p33","Estimate"]) / sqrt(12)
## And the measurement noise standard deviation
sqrt(exp(val$coefficients["e11","Estimate"]))


## ----sdlevels-----------------------------------------------------------------
val <- predict(fitTiTeTh)[[1]]
head(val$state$sd)


## ----selevels2----------------------------------------------------------------
valfilt <- filter.ctsmr(fitTiTeTh)[[1]]

tail(val$state$sd)
tail(valfilt$sd)

sqrt(tail(val$state$sd^2 - valfilt$sd^2))

