#run calibration experiments
#use last x-years for validation
#run the calibration several times with fixed sub-parameters
#my hypothesis is, that the location affects the estimate. do that for cherry, apricot, almond

#run calibration for single location and for combined location
#probably also regularization is also important


library(tidyverse)
library(DEoptim)
# #fix cultivar names with difficult characters
# diff_name <- c('Président Drouard', "Packham's Triumph", 'Dorée', 
#                'Búlida', 'Ramón Oliva', 'Precòce Bernard', 
#                'Napoleón', 'Ambrunés', 'Président Heron')
# corr_name <-  c('President Drouard', "Packhams Triumph", 'Doree', 
#                 'Bulida', 'Ramon Oliva', 'Precoce Bernard', 
#                 'Napoleon', 'Ambrunes', 'President Heron')
# pheno <- read.csv('phenology/adamedor_sub.csv', encoding = 'latin1') 
# for(i in 1:length(diff_name)){
#   pheno$cultivar[pheno$cultivar == diff_name[i]] <- corr_name[i]
# }
# unique(pheno$cultivar)
# write.csv(pheno, file = 'phenology/adamedor_sub.csv', row.names = FALSE)

pheno_df <- read.csv('phenology/adamedor_sub.csv', encoding = 'latin1')
#multiple location
spec.cult.mult <- pheno_df %>% 
  filter(year >= 1959) %>% 
  group_by(cultivar, species) %>% 
  summarise(nloc = length(unique(location))) %>% 
  ungroup() %>% 
  filter(nloc > 1) %>% 
  mutate(spec.cult = paste(species, cultivar, sep = '.')) %>% 
  pull(spec.cult)
  
pheno <- pheno_df %>% 
  filter(species %in% c('Sweet Cherry', 'Apple', 'Almond', 'Apricot')) %>% 
  group_by(species, cultivar, location) %>% 
  group_split()

nmin <- 10
share_cal <- 0.75

n <- purrr::map_int(pheno, nrow)
keep <- n > nmin
pheno <- pheno[keep]

#for each cultivar, sort by year, take 75% earliest observation and make them calibration
for(i in 1:length(pheno)){
  
  pheno[[i]] <- pheno[[i]] %>% 
    arrange(year)
  
  nobs <- nrow(pheno[[i]])
  n_cal <- floor(nobs * share_cal)
  
  pheno[[i]]$split <- 'validation'
  pheno[[i]]$split[1:n_cal] <- 'calibration'
  
  
}
names(pheno) <- purrr::map_chr(pheno, function(x) paste(x$species[1], x$cultivar[1], x$location[1], sep = '.'))

#combine entries with multiple location as an additional entry
mult.cult.list <- list()
for(i in 1:length(spec.cult.mult)){
  j <- grep(pattern = spec.cult.mult[i],x = names(pheno)) 
  mult.cult.list[[i]] <- do.call('rbind', pheno[j] ) 
}
names(mult.cult.list) <- paste(spec.cult.mult, 'combined', sep = '.')

pheno <- c(pheno, mult.cult.list)


eval_phenoflex_onlyreq  <- function(x,
                                    modelfn,
                                    bloomJDays,
                                    SeasonList,
                                    par_fixed,
                                    na_penalty = 365){
  
  
  par <- c(x, par_fixed)
  pred_bloom <- NULL
  
  pred_bloom <- unlist(lapply(X = SeasonList, FUN = modelfn, par = par))
  pred_bloom <- ifelse(is.na(pred_bloom), yes = na_penalty, no = pred_bloom)
  
  F <- sum((pred_bloom - bloomJDays)^2)
  return(F)
}

cka <- read.csv('weather_hourly/klein-altendorf_hourly.csv')
zaragoza <- read.csv('weather_hourly/zaragoza_hourly.csv')
santomera <- read.csv('weather_hourly/santomera_hourly.csv')
meknes <- read.csv('weather_hourly/meknes_hourly.csv')
sfax <- read.csv('weather_hourly/sfax_hourly.csv')
cieza <- read.csv('weather_hourly/cieza_hourly.csv')

weather_list <- list('Klein-Altendorf' = cka,
                     'Zaragoza' = zaragoza,
                     'Santomera' = santomera,
                     'Meknes' = meknes,
                     'Sfax' = sfax,
                     'Cieza' = cieza)

seasonlist <- purrr::map(1:length(weather_list), function(i){
  
  loc <- names(weather_list)[i]

  weather_list[[i]] %>% 
    chillR::genSeasonList(years = (min(weather_list[[i]]$Year)+1):max(weather_list[[i]]$Year)) %>% 
    stats::setNames(paste(loc, (min(weather_list[[i]]$Year)+1):max(weather_list[[i]]$Year), sep = '.')) %>% 
    return()
  
})

seasonlist <- unlist(seasonlist, recursive = FALSE)

rm(weather_list, cka, cieza, santomera, zaragoza, sfax, meknes)


#save both objects as RDS to run in on cluster
saveRDS(pheno, file = 'cluster/data/in/pheno.RDS')
saveRDS(seasonlist, file = 'cluster/data/in/seasonlist.RDS')

#create a task list for the cluster
data.frame(names(pheno), task = 1:length(names(pheno))) %>% 
  write.csv(file = 'cluster/data/in/task.csv', row.names = FALSE)



xU <- c(80, 500, 1.5)
xL <- c(10, 100, 0.1) 
par_fixed <- c(25, 4153.5, 12888.8, 139500, 2567000000000000000, 4, 36, 4, 1.6)


dir.create('parameter_phenoflex_classic')
fname <- 'parameter_phenoflex_classic/parameter_phenoflex.csv'

name_split <- str_split(names(pheno), pattern = '\\.')

#calibrate the models
i <- 1
for(j in 1:length(pheno)){
  
  for(r in 1:1){
    
    cat(names(pheno)[j], r, '\n')
    
    #skip entry if already calculated
    if(file.exists(fname)){
      res_df <- read.csv(fname, encoding = 'latin1')
      
      n <- res_df %>% 
        filter(repetition == r, 
               id == names(pheno)[j]) %>% 
        nrow()
      
      if(n != 0) next
    }
    
    pheno_sub <- pheno[[j]]$pheno[pheno[[j]]$split == 'calibration']
    yr_sub <- pheno[[j]]$year[pheno[[j]]$split == 'calibration']
    loc_yr <- paste(name_split[[j]][3], yr_sub, sep = '.')
    
    #make sure not to include seasons that are not covered
    keep <- loc_yr %in% names(seasonlist)
    loc_yr <- loc_yr[keep]
    pheno_sub <- pheno_sub[keep]

    set.seed(r)
    
    res <- DEoptim(fn = eval_phenoflex_onlyreq,
                   lower = xL,
                   upper = xU,
                   control = DEoptim.control(itermax = 200,
                                             parallelType = 1),
                   modelfn = evalpheno::custom_PhenoFlex_GDHwrapper,
                   bloomJDays = pheno_sub,
                   SeasonList = seasonlist[loc_yr],
                   par_fixed = par_fixed)
    
    par <- res$optim$bestmem
    par_df <- data.frame(id = names(pheno)[j],
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



