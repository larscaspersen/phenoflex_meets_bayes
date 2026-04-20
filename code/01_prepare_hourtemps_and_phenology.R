library(chillR)
library(tidyverse)
dir.create('code')

adamedor <- read.csv('../2023_adamedor-dataset_fitting/data/master_phenology_repeated_splits.csv') %>% 
  filter(repetition == 1)

dir.create('phenology')
write.csv(adamedor, file = 'phenology/adamedor_sub.csv', row.names = FALSE)

stations <- read.csv('../2023_adamedor-dataset_fitting/data/weather_ready/weather_station_phenological_observations.csv')  %>% 
  mutate(id = station_name,
         Latitude = latitude,
         Longitude = longitude)

#for each weather station get the median year, then make the baseline adjustment
cka <- read.csv('../2023_adamedor-dataset_fitting/data/weather_ready/cka_clean.csv') %>% 
  filter(Year < 2022)
cieza <- read.csv('../2023_adamedor-dataset_fitting/data/weather_ready/cieza_clean_patched.csv')
sfax <- read.csv('../2023_adamedor-dataset_fitting/data/weather_ready/sfax_clean.csv')
meknes <- read.csv('../2023_adamedor-dataset_fitting/data/weather_ready/meknes_clean.csv')
zaragoza <- read.csv('../2023_adamedor-dataset_fitting/data/weather_ready/zaragoza_clean.csv') %>% 
  filter(Year < 2022)
santomera <- read.csv('../2023_adamedor-dataset_fitting/data/weather_ready/murcia_clean.csv')

weather_obs_list <- list('Klein-Altendorf' = cka,
                         'Zaragoza' = zaragoza,
                         'Cieza' = cieza,
                         'Santomera' = santomera,
                         'Meknes' = meknes,
                         'Sfax' = sfax)
rm(zaragoza, cieza, cka, santomera, sfax, meknes)

dir.create('weather_hourly')

for(i in 1:nrow(stations)){
  
  fname <- paste0('weather_hourly/', tolower(stations$id[i]), '_hourly.csv')
  
  #if(file.exists(fname)) next
  
  weather_obs_list[[stations$id[i]]] %>% 
    stack_hourly_temps(latitude = stations$Latitude[i]) %>% 
    purrr::pluck('hourtemps') %>% 
    mutate(Temp = round(Temp, digits = 3)) %>% 
    select(Year, Month, Day, Hour, JDay, Temp) %>% 
    write.csv(file = fname, row.names = FALSE)
}
