#check files, list files that are not complete
run_cluster <- TRUE
nrep <- 10

if(run_cluster){
  .libPaths(c("~/Julians-R-packages/my-R-packages", .libPaths()))
}

path_in <- 'cluster/data/in/'
path_out <- 'cluster/data/out/'

if(run_cluster){
  
  path_in <- 'data/cal_hierac_pheno/in/'
  path_out <- 'data/cal_hierac_pheno/out/'
  
}


task_df <- read.csv(paste0(path_in, 'task.csv'))

fnames <- list.files(path_out)
fnames_full <- list.files(path_out, full.names = TRUE)

n_complete <-  sapply(fnames_full, FUN = function(x){
  xfile <- read.csv(x)
  return(max(xfile$repetition))
})
id <-  sapply(fnames_full, FUN = function(x){
  xfile <- read.csv(x)
  return(unique(xfile$id))
})

incomplete <- n_complete < nrep
task_incomplete <- task_df$task[task_df$names.pheno. %in%  id[incomplete]]

missing <- (task_df$names.pheno. %in% id) == FALSE
task_missing <- task_df$task[missing]

task_open <- sort(c(task_incomplete, task_missing))

#function to format ranges
format_ranges <- function(x) {
  x <- sort(unique(x))  # ensure sorted, no duplicates
  
  # identify breaks in consecutiveness
  breaks <- c(TRUE, diff(x) != 1)
  groups <- cumsum(breaks)
  
  # split into consecutive runs
  runs <- split(x, groups)
  
  # format each run
  parts <- sapply(runs, function(r) {
    if (length(r) == 1) {
      as.character(r)
    } else {
      paste0(min(r), ":", max(r))
    }
  })
  
  paste(parts, collapse = ",")
}

print(format_ranges(task_open))