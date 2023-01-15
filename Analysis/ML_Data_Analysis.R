# Machine Learning Stats Interpretation

tsv.setA <- read.table("E:/Taf/Machine_Learning/SetA/All_patients_Training_SetA.tsv", sep = "\t")
str(tsv.setA)

tsv.setB <- read.table("E:/Taf/Machine_Learning/SetB/All_patients_Training_SetB.tsv", sep = "\t")
str(tsv.setB)

tsv.setA.processed <- read.table("E:/Taf/Machine_Learning/SetA/All_patients_PP_T_SetA.tsv", sep = ",")

tsv.setB.processed <- read.table("E:/Taf/Machine_Learning/SetB/All_patients_PP_T_SetB.tsv", sep = ",") 

get.headers <- read.table("E:/Taf/Machine_Learning/SetB/Training_SetB/p100001.psv", sep ="|", header = TRUE)
get.headers <- names(get.headers)

names(tsv.setA) <- get.headers
names(tsv.setB) <- get.headers

get.headers <- get.headers[get.headers != "EtCO2"]
names(tsv.setA.processed) <- get.headers
names(tsv.setB.processed) <- get.headers

# Boxplots

par(cex=3) # is for x-axis
par(col.axis="grey")
par(mgp = c(3,1,2))

png(filename="Boxplot_SetA.png", 1600, 1600)
bp <- boxplot(tsv.setA, col = c("lightblue", "cyan"), xaxt = "n")
tick <- seq_along(bp$names)
axis(1, at = tick, labels = FALSE)
text(tick, par("usr")[3] - 300, bp$names, srt = 45, xpd = TRUE)
dev.off()

png(filename="Boxplot_SetA_processed.png", 1600, 1600)
bp <- boxplot(tsv.setA.processed, col = c("lightblue", "cyan"), xaxt = "n")
tick <- seq_along(bp$names)
axis(1, at = tick, labels = FALSE)
text(tick, par("usr")[3] - 0.75, bp$names, srt = 45, xpd = TRUE)
dev.off()

png(filename="Boxplot_SetB.png", 1600, 1600)
bp <- boxplot(tsv.setB, col = c("lightblue", "cyan"), xaxt = "n")
tick <- seq_along(bp$names)
axis(1, at = tick, labels = FALSE)
text(tick, par("usr")[3] - 300, bp$names, srt = 45, xpd = TRUE, offset = 20)
dev.off()

png(filename="Boxplot_SetB_processed.png", 1600, 1600)
bp <- boxplot(tsv.setB.processed, col = c("lightblue", "cyan"), xaxt = "n")
tick <- seq_along(bp$names)
axis(1, at = tick, labels = FALSE)
text(tick, par("usr")[3] - 0.75, bp$names, srt = 45, xpd = TRUE, offset = 20)
dev.off()
