data = read.csv('C:/Users/vyasy/Desktop/Data/adv_sales.csv')
plot(data$Advertising, type = "l" , ylim=c(0,80),col = "red" , ylab="Expenditure", xlab = 'Time')
lines(data$Sales,col = "blue")
legend(30,80, lty = 1,col = c("red","blue"),c("Adv","Sales"))
par(mfrow= c(1,1))
with(data,acf(Advertising,lag.max = 36))
with(data,acf(Sales,lag.max = 36))

with(data,ccf(Advertising,Sales,lag.max = 36,ylab = "CCF ", main = "Adv vs Sales"))
with(data,ccf(ar(Advertising)$resid,ar(Sales)$resid,lag.max = 36,na.action = na.pass,ylab = "CCF",
              main = "RESI Adv vs Sales"))
library(vars)

fitvar1 <- vars::VAR(data[,2:3], p =1, type=c("both"))
fitvar1

summary(fitvar1,equations = "Sales")
plot(fitvar1,names = "Sales")
 