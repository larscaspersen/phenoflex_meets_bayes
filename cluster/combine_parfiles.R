#check files, list files that are not complete
run_cluster <- TRUE

if(run_cluster){
  .libPaths(c("~/Julians-R-packages/my-R-packages", .libPaths()))
}

path_out <- 'cluster/data/out/'

if(run_cluster){
  
  path_out <- 'data/calibration_hierach_model/out/'
  
}

fnames <- list.files(path_out, full.names = TRUE)

flist <- lapply(fnames, FUN = read.csv)
fdf <- do.call('rbind', flist)

write.csv(flist, file = paste0(path_out, 'par-onlyreq.all.csv'), row.names = FALSE)
