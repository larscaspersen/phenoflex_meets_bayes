run_cluster <- FALSE
nrep <- 10

if(run_cluster){
  .libPaths(c("~/Julians-R-packages/my-R-packages", .libPaths()))
  library(optparse)
}
library(DEoptim)
library(tidyverse)
library(chillR)

path_in <- 'cluster/data/in/'
path_out <- 'cluster/data/out/'
path_code <- 'cluster/helpers_calibration.R'

job_id <- 46
if(run_cluster){
  
  path_in <- 'data/cal_hierac_pheno/in/'
  path_out <- 'data/cal_hierac_pheno/out/'
  path_code <- 'code/calibration_hierach_model/helpers_calibration.R'
  
  #read the jobid
  option_list <- list(make_option("--job-id", type="integer"))
  opt <- parse_args(OptionParser(option_list=option_list))
  job_id <- opt$"job-id"
}

source(path_code)

#read task
task_df <- read.csv(paste0(path_in, 'task.csv')) %>% 
  filter(task == job_id)

#read season and pheno
pheno <- readRDS(paste0(path_in, 'pheno.RDS'))

season <- readRDS(paste0(path_in, 'seasonlist.RDS'))


xU <- c(80, 500, 1.5)
xL <- c(10, 100, 0.1) 
par_fixed <- c(25, 4153.5, 12888.8, 139500, 2567000000000000000, 4, 36, 4, 1.6)

#split names, to extract species, cultivar, location(s)
name_split <- str_split(task_df$names.pheno., pattern = '\\.')

#initialize object
pheno_sub <- data.frame()

#calibrate the models
j <- 1
r<- 1
for(j in 1:nrow(task_df)){
  
  cat('---------\n')
  cat(task_df$names.pheno.[j], '\n')
  
  #subset pheno
  pheno_sub <- pheno[[task_df$names.pheno.[j]]]
  pheno_sub$loc.year <- paste(pheno_sub$location, pheno_sub$year, sep = '.')
  use_for_cal <- pheno_sub$split == 'calibration'
  
  #make sure not to include seasons that are not covered
  keep <- pheno_sub$loc.year %in% names(season)
  pheno_sub <- pheno_sub[keep,]
  
  
  #generate file name
  fname <- paste0(path_out, 'par_onlyreq_', 
                      gsub(pattern = ' ', replacement = '',
                           task_df$names.pheno[j]), '.csv')
  
  for(r in 1:nrep){
    
    cat(r, 'of', nrep, '\n')
    
    #skip entry if already calculated
    if(file.exists(fname)){
      res_df <- read.csv(fname)
      
      n <- res_df %>% 
        filter(repetition == r, 
               id == task_df$names.pheno.[j][j]) %>% 
        nrow()
      
      if(n != 0) next
    }

    set.seed(r)
    
    res <- DEoptim(fn = eval_phenoflex_onlyreq,
                   lower = xL,
                   upper = xU,
                   control = DEoptim.control(itermax = 200),
                   # control = DEoptim.control(itermax = 200,
                   #                           parallelType = 1),
                   modelfn = custom_PhenoFlex_GDHwrapper,
                   bloomJDays = pheno_sub$pheno[use_for_cal],
                   SeasonList = season[pheno_sub$loc.year[use_for_cal]],
                   par_fixed = par_fixed)
    
    par <- res$optim$bestmem
    par_df <- data.frame(id = task_df$names.pheno.[j],
                         repetition = r,
                         yc = par[1],
                         zc = par[2],
                         s1 = par[3],
                         Tu = par_fixed[1],
                         E0 = par_fixed[2],
                         E1 = par_fixed[3],
                         A0 = par_fixed[4],
                         A1 = par_fixed[5],
                         Tf = par_fixed[6],
                         Tc = par_fixed[7],
                         Tb = par_fixed[8],
                         slope = par_fixed[9],
                         timestamp = Sys.time())
    
    
    #flag if the table is appended or newly created
    append_file <- FALSE
    add_colnames <- TRUE
    if(file.exists(fname)){
      append_file <- TRUE
      add_colnames <- FALSE
    } 
    write.table(par_df, file = fname, row.names = FALSE, append = append_file, sep = ',', col.names = add_colnames)
    
  }
}


