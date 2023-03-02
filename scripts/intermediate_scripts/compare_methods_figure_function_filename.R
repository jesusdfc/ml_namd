library(ggplot2)
library(gridExtra)
args = commandArgs(trailingOnly=TRUE)


#LOAD THE DATASET
Model <- args[1]
message('Loading file with Model ',Model)
excel_df <- read.table(file=paste0('inputs',Model,'_best_conf_6_6.tsv'), sep='\t', header = TRUE, row.names = NULL)

#------------------------->  LETS GENERATE SOME TABLES <------------------------#

message('Generating tables')
table_df <- aggregate(excel_df[['Val.BalAcc']], excel_df[c('FileName','Model')],mean)
colnames(table_df) <- c('Variables','Model', 'Mean ValBAC')

#ADD BALANCED ACCURACY
table_df['Max ValBAC'] <- list('Max ValBAC' = aggregate(excel_df[['Val.BalAcc']], excel_df[c('FileName','Model')],max)$x)
table_df['Min ValBAC'] <- list('Min ValBAC' = aggregate(excel_df[['Val.BalAcc']], excel_df[c('FileName','Model')],min)$x)

#ADD AUAC
table_df['Mean ValAUAC'] <- list('Mean ValAUAC' = aggregate(excel_df[['Val.AUAC']], excel_df[c('FileName','Model')],mean)$x)
table_df['Min ValAUAC'] <- list('Min ValAUAC' = aggregate(excel_df[['Val.AUAC']], excel_df[c('FileName','Model')],min)$x)
table_df['Max ValAUAC'] <- list('Max ValAUAC' = aggregate(excel_df[['Val.AUAC']], excel_df[c('FileName','Model')],max)$x)

#ADD AURC
table_df['Mean ValAURC'] <- list('Mean ValAURC' = aggregate(excel_df[['Val.AURC']], excel_df[c('FileName','Model')],mean)$x)
table_df['Min ValAURC'] <- list('Min ValAURC' = aggregate(excel_df[['Val.AURC']], excel_df[c('FileName','Model')],min)$x)
table_df['Max ValAURC'] <- list('Max ValAURC' = aggregate(excel_df[['Val.AURC']], excel_df[c('FileName','Model')],max)$x)

#ADD AUROC
table_df['Mean ValAUROC'] <- list('Mean ValAUROC' = aggregate(excel_df[['Val.AUROC']], excel_df[c('FileName','Model')],mean)$x)
table_df['Min ValAUROC'] <- list('Min ValAUROC' = aggregate(excel_df[['Val.AUROC']], excel_df[c('FileName','Model')],min)$x)
table_df['Max ValAUROC'] <- list('Max ValAUROC' = aggregate(excel_df[['Val.AUROC']], excel_df[c('FileName','Model')],max)$x)


#ADD Accuracy
table_df['Mean ValAcc'] <- list('Mean ValAcc' = aggregate(excel_df[['Val.Accuracy']], excel_df[c('FileName','Model')],mean)$x)
table_df['Min ValAcc'] <- list('Min ValAcc' = aggregate(excel_df[['Val.Accuracy']], excel_df[c('FileName','Model')],min)$x)
table_df['Max ValAcc'] <- list('Max ValAcc' = aggregate(excel_df[['Val.Accuracy']], excel_df[c('FileName','Model')],max)$x)

#REFORMAT VARIABLES
table_df[['Variables']] <- sapply(table_df[['Variables']],function(x) substr(as.character(x),1,nchar(as.character(x))-7))
#Reduce it to only contain the variables that have not been dropped
table_df[['Variables']] <- sapply(table_df[['Variables']],function(x){
  
  split_x <- stringr::str_split(x,'_')[[1]]
  
  new_x <- c()
  for (i in seq_along(split_x)){
    
    if (i%%2){
      
      if (split_x[i+1]=='False'){
        new_x <- c(new_x,split_x[i])
      }
      
    }
    
  }
  
  return(paste0(new_x,collapse='_'))
  
})


#Order by MAX ValBAC
table_df <- table_df[order(table_df$'Max ValBAC',decreasing=TRUE),]

pdf(paste0('figures/',Model,'table_summary_orderby_valBAC.pdf'),height = 140, width = 30)
grid.table(as.data.frame(table_df),rows=NULL)
dev.off()

#Order by MAX ValAUROC
table_df <- table_df[order(table_df$'Max ValAUROC',decreasing=TRUE),]

pdf(paste0('figures/',Model,'table_summary_orderby_valAUROC.pdf'),height = 140, width = 30)
grid.table(as.data.frame(table_df),rows=NULL)
dev.off()
