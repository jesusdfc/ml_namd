library(ggpubr)
library(MASS)
library(reshape2)
library(reshape)

dfm <- read.delim(paste0(getwd(),'/inputs','/allele_frequencies.tsv'))
colnames(dfm) <- c("Gene", "SNP", "Minor.Allele", "Total Alelles MAC",
                   "NonAtrophy 36m Alleles MAC", "Atrophy 36m Alleles MAC",
                   "Allele Atrophy Uncorrected Pvalue",
                   "NonFibrosis 36m Alleles MAC", "Fibrosis 36m Alleles MAC",
                   "Allele Fibrosis Uncorrected Pvalue")

# First drop values that are not percentage
dfm_percentage <- dfm[-seq(1, nrow(dfm), 2),]
dfm_percentage_atrophy <- dfm_percentage[,c(1,2,3,4,5,6,7)]
dfm_percentage_fibrosis <-dfm_percentage[,c(1,2,3,4,8,9,10)]

#### ATROPHY
#make the atrophy to have negative percentages
dfm_percentage_atrophy["Atrophy 36m Alleles MAC"] <- -as.integer(dfm_percentage_atrophy[["Atrophy 36m Alleles MAC"]])
#keep only the names we want to keep
dfm_percentage_atrophy_plot <- dfm_percentage_atrophy[,c(2,3,5,6)]
dfm_percentage_atrophy_plot_melted <- melt(dfm_percentage_atrophy_plot, id = c('SNP', 'Minor.Allele'))
dfm_percentage_atrophy_plot_melted[['value']] <- as.integer(dfm_percentage_atrophy_plot_melted[['value']])
colnames(dfm_percentage_atrophy_plot_melted) <- c('SNP','Minor Allele','Type','Alelle Frequency (%)')
dfm_percentage_atrophy_plot_melted[['SNP']] <- paste(dfm_percentage_atrophy[['Gene']],
                                                      dfm_percentage_atrophy[['SNP']])

#save as tsv
#write.table(dfm_percentage_atrophy_plot_melted, 'atrophy_36m_alleles_frequency.tsv',sep='\t')



#### FIBROSIS
#make the fibrosis to have negative percentages
dfm_percentage_fibrosis["Fibrosis 36m Alleles MAC"] <- -as.integer(dfm_percentage_fibrosis[["Fibrosis 36m Alleles MAC"]])
#keep only the names we want to keep
dfm_percentage_fibrosis_plot <- dfm_percentage_fibrosis[,c(2,3,5,6)]
dfm_percentage_fibrosis_plot_melted <- melt(dfm_percentage_fibrosis_plot, id = c('SNP', 'Minor.Allele'))
dfm_percentage_fibrosis_plot_melted[['value']] <- as.integer(dfm_percentage_fibrosis_plot_melted[['value']])
colnames(dfm_percentage_fibrosis_plot_melted) <- c('SNP','Minor Allele','Type','Alelle Frequency (%)')
dfm_percentage_fibrosis_plot_melted[['SNP']] <- paste(dfm_percentage_fibrosis[['Gene']],
                                                     dfm_percentage_fibrosis[['SNP']])

#save as tsv
#write.table(dfm_percentage_fibrosis_plot_melted, 'fibrosis_36m_alleles_frequency.tsv',sep='\t')



# ggbarplot(dfm_percentage_atrophy_plot_melted,
#           x = "SNP", y = "Alelle Frequency (%)",
#           fill = "Type",           # change fill color by mpg_level
#           color = "white",            # Set bar border colors to white
#           palette = "jco",            # jco journal color palett. see ?ggpar
#           sort.val = "desc",          # Sort the value in descending order
#           sort.by.groups = FALSE,     # Don't sort inside each group
#           x.text.angle = 90,          # Rotate vertically x axis texts
#           ylab = "MPG z-score",
#           legend.title = "MPG Group",
#           rotate = TRUE,
#           ggtheme = theme_minimal()
# )
