args = commandArgs(trailingOnly=TRUE)
initequity = strtoi(args[1])

oldw <- getOption("warn")
options(warn = -1)

# Check if packages are installed (and install if not)
packages = c("BatchGetSymbols", "dplyr", "tidyverse", "quantstrat")

# Now load or install & load all
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)

initdate <-  "2001-01-01"   # Set Dates and Info
from <- "2016-01-01"
to <- "2021-12-31" # This will give us 5 years of trading data for the selected stocks
currency ("USD")
Symbols <- c("TSLA", "NVDA", "WST", "PAYC", "ODFL")
stock(Symbols,currency='USD',multiplier=1)
Sys.setenv (TZ = "UTC")

# Get Data
getSymbols(Symbols, from = from,
           to = to, src = "yahoo",
           adjust = TRUE,
           index.class=c("POSIXt","POSIXct"))

# Data Processesing and Model Building
Scale_Me <- function(x){
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}
TS_preprocess <- function(dat){
  dat <- data.frame(dat)
  colnames(dat) <- c("open", "high", "low", "close", "volume", "adjusted")
  dat$Y <- with(dat, ifelse(close >= open, 1, 0))
  dat$X1 <- SMA(lag(dat$close), n = 10)
  dat$X2 <- RSI(lag(dat$close), nFast = 14, nSlow = 26, nSig = 9, maType = SMA)
  dat$X3 <- momentum(lag(dat$close), n = 12)
  dat <- dat[complete.cases(dat), ]
  #dat = cbind(dat[, 'Y'], apply(dat[, 8:ncol(dat)], 2, Scale_Me))
  #colnames(dat)[1] = "Y"
  dat <- dat[, c("Y", "X1", "X2", "X3")]
  dat <- as.xts(dat)
  return(dat)
}
df.NVDA <- TS_preprocess(NVDA)
df.TSLA <- TS_preprocess(TSLA)
df.WST <- TS_preprocess(WST)
df.PAYC <- TS_preprocess(PAYC)
df.ODFL <- TS_preprocess(ODFL)

n_train <- 100
n_test <- 1

LogistFun <- function(frm, dat, trainIndex, testIndex){
  LogitModel <- glm(frm, data = dat[trainIndex, ])
  pred <- predict(LogitModel, newdata = dat[testIndex, ], type = 'response')
  return(pred)
}

RollingBacktest <- function(dat, ntrain = n_train, ntest = n_test){
  stopifnot('Y' %in% names(dat))
  frm_ <- formula(reformulate(paste0("X", seq(2:ncol(dat))), "Y"))

  stride <- ntrain + ntest
  startPosn <- seq(1, dim(dat)[1] - stride)

  train_index_list <- lapply(startPosn, function(i) seq(i, i + ntrain))
  test_index_list <- lapply(startPosn, function(i) seq((i + ntrain + 1), (i + ntrain + ntest)))

  mapply(LogistFun, trainIndex = train_index_list, testIndex = test_index_list, MoreArgs = list(frm = frm_, dat = dat), SIMPLIFY = FALSE
  )
}

TS_postprocess <- function(dat, ntrain){
  results <- tail(dat, -(ntrain + 1))
  results$probs <- RollingBacktest(dat)
  results$predictions <- ifelse(results$probs > 0.75, 1, 0)
#    print(paste0("Model Accuracy at the 0.60 prob cut-off ", mean(results$Y == results$predictions)))
  return(results)
}
out.NVDA <- TS_postprocess(df.NVDA, ntrain = n_train)
out.WST <- TS_postprocess(df.WST, ntrain = n_train)
out.TSLA <- TS_postprocess(df.TSLA, ntrain = n_train)
out.ODFL <- TS_postprocess(df.ODFL, ntrain = n_train)
out.PAYC <- TS_postprocess(df.PAYC, ntrain = n_train)

out.NVDA <- na.omit(cbind(NVDA, out.NVDA))
out.WST <- na.omit(cbind(WST, out.WST))
out.TSLA <- na.omit(cbind(TSLA, out.TSLA))
out.ODFL <- na.omit(cbind(ODFL, out.ODFL))
out.PAYC <- na.omit(cbind(PAYC, out.PAYC))

NVDA <- out.NVDA
WST <- out.WST
TSLA <- out.TSLA
ODFL <- out.ODFL
PAYC <- out.PAYC

stock("NVDA", currency = "USD", multiplier = 1)
stock("WST", currency = "USD", multiplier = 1)
stock("TSLA", currency = "USD", multiplier = 1)
stock("PAYC", currency = "USD", multiplier = 1)
stock("ODFL", currency = "USD", multiplier = 1)

# Prepare Portfolio and Account
initeq <- initequity # Our initial equity is now dynamic
tradesize <- .02*initeq # Every trade we will risk 2% of our equity (money management)
# Note: I tried to set tradesize dynamically (2% of current equity), but I couldn't get it

strategy.st <- portfolio.st <- account.st <- "firststrat"
rm.strat(strategy.st) # Remove existing strategy
initPortf(portfolio.st,
          symbols = Symbols,
          initDate = initdate,
          currency = "USD") # Initialize portfolio
initAcct(account.st,
         portfolios = portfolio.st,
         initDate = initdate,
         currency = "USD",
         initEq = initeq) # Initialize account
initOrders(portfolio.st, symbols = Symbols, initDate = initdate)
strategy(strategy.st, store = TRUE)

# Add Indicators
add.indicator(strategy = strategy.st,
              name = "SMA",
              arguments = list(x = quote(Cl(mktdata)), n = 200),
              label = "SMA200")

add.indicator(strategy = strategy.st,
              name = "SMA",
              arguments = list(x = quote(Cl(mktdata)), n = 50),
              label = "SMA50")


# Add Signals
nMult_orderqty <- 2
addPosLimit(portfolio.st, symbol = "NVDA", timestamp = initdate, maxpos = nMult_orderqty * tradesize)
addPosLimit(portfolio.st, symbol = "WST", timestamp = initdate, maxpos = nMult_orderqty * tradesize)
addPosLimit(portfolio.st, symbol = "TSLA", timestamp = initdate, maxpos = nMult_orderqty * tradesize)
addPosLimit(portfolio.st, symbol = "ODFL", timestamp = initdate, maxpos = nMult_orderqty * tradesize)
addPosLimit(portfolio.st, symbol = "PAYC", timestamp = initdate, maxpos = nMult_orderqty * tradesize)

add.signal(strategy = strategy.st,
           name = "sigThreshold",
           arguments = list(threshold = 0.75,
                            column = "probs",
                            relationship = "gt",
                            cross = TRUE),
           label = "longSig")

add.signal(strategy = strategy.st,
           name = "sigThreshold",
           arguments = list(threshold = 0.25,
                            column = "probs",
                            relationship = "lt",
                            cross = TRUE),
           label = "exitlongSig")

add.signal(strategy.st,
           name = "sigCrossover",
           arguments = list(columns = c("SMA50", "SMA200"),
                            relationship = "gt"),
           label = "longfilter")

add.signal(strategy.st,
           name = "sigComparison",
           arguments = list(columns = c("SMA50", "SMA200"),
                            relationship = "lt" ),
           label = "filterexit")


# Add Entry Rules
add.rule(strategy.st, name = "ruleSignal",
         arguments = list(sigcol = "longfilter", sigval = TRUE,
                          orderqty = tradesize, ordertype = "market",
                          orderside = "long", replace = FALSE,
                          prefer = "Open"),
         type = "enter")

add.rule(strategy = strategy.st,
         name = "ruleSignal",
         arguments = list(sigcol = "longSig", sigval = 1,
                          orderqty = tradesize, ordertype = "market",
                          orderside = "long", osFUN = osMaxPos,
                          prefer = "Open", replace = TRUE),
         type = "enter",
         label = "EnterLONG")

# Add Exit Rules
add.rule(strategy.st,
         name = "ruleSignal",
         arguments = list(sigcol = "exitlongSig", sigval = 1,
                          orderqty = "all", ordertype = "market",
                          orderside = "long", osFUN = osMaxPos,
                          prefer = "Open", replace = TRUE),
         type = "exit",
         label = "ExitLong")

add.rule(strategy.st, name = "ruleSignal",
         arguments = list(sigcol = "filterexit", sigval = TRUE,
                          orderqty = "all", ordertype = "market",
                          orderside = "long", replace = FALSE,
                          prefer = "Open"),
         type = "exit")

# Apply the Strategy:
applyStrategy(strategy.st, portfolios = portfolio.st)
updatePortf(portfolio.st)
updateAcct(account.st)
updateEndEq(account.st)

# Plot the Results:
chart.Posn(portfolio.st, Symbol = "NVDA")
chart.Posn(portfolio.st, Symbol = "TSLA")
chart.Posn(portfolio.st, Symbol = "WST")
chart.Posn(portfolio.st, Symbol = "PAYC")
chart.Posn(portfolio.st, Symbol = "ODFL")


################ Trade Statistics & Expectunity #######################################
perTrade <- perTradeStats(portfolio.st,"NVDA")
tstats <- tradeStats(Portfolios = portfolio.st)

# Expectunity Function
Expectunity <- function(perTrade,tstats){
  pldata <- data.frame(perTrade$Start,perTrade$End,perTrade$Net.Trading.PL)
  pldata2 <- cbind(pldata,tstats$Num.Trades,abs(tstats$Avg.Losing.Trade),abs(tstats$Largest.Loser))
  pldata2 <- pldata2 %>%
    rename(
      Start = perTrade.Start,
      End = perTrade.End,
      Net.PL = perTrade.Net.Trading.PL,
      Num.Trades = 'tstats$Num.Trades',
      Avg.Loss = 'abs(tstats$Avg.Losing.Trade)',
      Large.Loss = 'abs(tstats$Largest.Loser)'
    )
  pldata2$R.Mult1 <- pldata2$Net.PL/pldata2$Avg.Loss
  pldata2$R.Mult2 <- pldata2$Net.PL/pldata2$Large.Loss
  StrategyDays <- tail(pldata2$Start,n=1)-head(pldata2$Start,n=1)
  Expectancy.Rmult1 <- sum(pldata2$R.Mult1/pldata2$Num.Trades)
  Expectancy.Rmult2 <- sum(pldata2$R.Mult2/pldata2$Num.Trades)
  Opportunities <- pldata2$Num.Trades[1]*365/as.numeric(StrategyDays)
  StdDev.Rmult1 <- sd(pldata2$R.Mult1)
  Expectunity.Rmult1 <- Expectancy.Rmult1*Opportunities
  Expectunity.Rmult2 <- Expectancy.Rmult2*Opportunities
  System_Quality <- Expectancy.Rmult1/StdDev.Rmult1*sqrt(pldata2$Num.Trades[1])
  df <- data.frame(round(Expectunity.Rmult1,2), round(Expectunity.Rmult2,2),
                   round(tstats$Num.Trades),round(tstats$Net.Trading.PL,2),
                   round(System_Quality,2),round(tstats$Percent.Positive,2))
  colnames(df) <- c("Expectunity, R1","Expectunity, R2", "Num Trades",
                    "Net Profit","System Quality", "Perc. Positive")
  return(df)
}

# Expectunity for NVDA:
pT.NVDA <- perTradeStats(portfolio.st,"NVDA")
ts.NVDA <- tradeStats(portfolio.st,"NVDA")
NVDA.info <- Expectunity(pT.NVDA, ts.NVDA)

# Expectunity for TSLA:
pT.TSLA <- perTradeStats(portfolio.st,"TSLA")
ts.TSLA <- tradeStats(portfolio.st,"TSLA")
TSLA.info <- Expectunity(pT.TSLA, ts.TSLA)

# Expectunity for WST:
pT.WST <- perTradeStats(portfolio.st,"WST")
ts.WST <- tradeStats(portfolio.st,"WST")
WST.info <- Expectunity(pT.WST, ts.WST)

# Expectunity for PAYC:
pT.PAYC <- perTradeStats(portfolio.st,"PAYC")
ts.PAYC <- tradeStats(portfolio.st,"PAYC")
PAYC.info <- Expectunity(pT.PAYC, ts.PAYC)

# Expectunity for ODFL:
pT.ODFL <- perTradeStats(portfolio.st,"ODFL")
ts.ODFL <- tradeStats(portfolio.st,"ODFL")
ODFL.info <- Expectunity(pT.ODFL, ts.ODFL)

stock.info <- rbind(NVDA.info,TSLA.info,WST.info,PAYC.info,ODFL.info)
rownames(stock.info) <- c("NVDA","TSLA","WST","PAYC","ODFL")


# System Averages:
Exp1.Average <- mean(stock.info$'Expectunity, R1') # System expectunity
Exp2.Average <- mean(stock.info$'Expectunity, R2')
NumTrades.Average <- mean(stock.info$`Num Trades`)
NumTrades.Total <- sum(stock.info$`Num Trades`)
NetProf.Total <- sum(stock.info$`Net Profit`)
SysQual.Average <- mean(stock.info$`System Quality`) # System quality
PercPos.Average <- mean(stock.info$`Perc. Positive`)

# Per-Trade Info:
pT.NVDA$Stock <- c("NVDA")
pT.TSLA$Stock <- c("TSLA")
pT.WST$Stock <- c("WST")
pT.PAYC$Stock <- c("PAYC")
pT.ODFL$Stock <- c("ODFL")
perTrade.all <- rbind(pT.NVDA,pT.TSLA,pT.WST,pT.PAYC,pT.ODFL) # Info per trade

# Trade-Stats info:
tradeStats.all <- rbind(ts.NVDA,ts.TSLA,ts.WST,ts.PAYC,ts.ODFL) # Info on trade stats
tradeStats.all <- cbind(tradeStats.all,stock.info$'Expectunity, R1')
colnames(tradeStats.all)[33] <- "Expectunity"

options(warn = oldw)

write.csv(tradeStats.all, "C:/Users/rgjoh/Downloads/tradeStats.csv")