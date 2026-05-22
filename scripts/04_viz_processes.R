#vizualize the parameters the mcmc obtained

get_transition_fun <- function(yc, s1, y_vector = NULL, p = 0.5){
  calc_py <- function(yc, s1, y){
    sy <- exp(s1 * yc * ((y-yc)/y))
    return((sy)/(sy+1))
  }
  
  if(is.null(y_vector)){
    y_vector <- seq(yc * (1-p), yc *(1+p), 1)
  }
  
  purrr::map_dbl(y_vector, function(y) calc_py(yc, s1, y))
  
}

hourtemps <- read.csv('KA_hourtemps.csv') %>% 
  chillR::genSeasonList(years = 1999:2005)

out <- chillR::PhenoFlex(temp = hourtemps[[1]]$Temp,
                         times = 1:length(hourtemps[[1]]$Temp),
                         E0 = 4153.5,
                         E1 = 12888.8,
                         A0 = 139500,
                         A1 = 2567000000000000000,
                         slope = 1.6,
                         yc = -3.4,
                         s1 = 0.57, 
                         zc = 99.47, basic_output = FALSE)

out_normal <- chillR::PhenoFlex(temp = hourtemps[[1]]$Temp,
                         times = 1:length(hourtemps[[1]]$Temp),
                         E0 = 4153.5,
                         E1 = 12888.8,
                         A0 = 139500,
                         A1 = 2567000000000000000,
                         slope = 1.6,
                         yc = 75,
                         s1 = 0.57, 
                         zc = 220, basic_output = FALSE)


viz_processes_phenoflex <- function(pheno_out,
                                    temp_df,
                                    yc,
                                    s1,
                                    xlim = NULL,
                                    ylim_temp = NULL,
                                    ylim_chill = NULL,
                                    ylim_heat = NULL, 
                                    ylim_py = NULL,
                                    annotate_subplot = FALSE){
  
  end_year <- max(temp_df$Year)


  
  #plot for temperature
  p_temp <- temp_df %>% 
    group_by(JDay, Year) %>% 
    summarise(Tmean = mean(Temp)) %>% 
    ungroup() %>% 
    mutate(run_mean = chillR::runn_mean(vec = Tmean, runn_mean = 15),
           yday_plot = ifelse(Year == end_year, yes = JDay, no = JDay - 365)) %>% 
    ggplot(aes(x=yday_plot)) +
    geom_line(aes(y = run_mean)) +
    geom_vline(xintercept = bloom_jday, linetype = 'dashed', col = 'black') +
    xlab('Date') +
    ylab('Daily Mean\nTemperature (°C)') +
    scale_x_continuous(breaks = c(275-365, 306-365, 336-365, 1, 32, 61, 92, 122), 
                       labels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May')) +
    coord_cartesian(xlim = xlim,
                    ylim = ylim_temp) +
    theme_bw(base_size = 13) +
    theme(axis.ticks.x = element_blank(),
          axis.title.x = element_blank(),
          axis.text.x = element_blank())
  
  pheno_out_df <- data.frame(
    yday_plot = ifelse(temp_df$Year == end_year, yes = temp_df$JDay, no = temp_df$JDay - 365),
    y = pheno_out$y,
    z = pheno_out$z,
    Py = get_transition_fun(yc = yc, s1 = s1, y_vector = pheno_out$y)
  )
  
  zc = NULL
  bloom_jday = NULL
  
  if(is.na(pheno_out$bloomindex) == FALSE){
    pheno_out_df$y[pheno_out$bloomindex:nrow(pheno_out_df)] <- NA
    pheno_out_df$z[pheno_out$bloomindex:nrow(pheno_out_df)] <- NA
    pheno_out_df$Py[pheno_out$bloomindex:nrow(pheno_out_df)] <- NA
    
    bloom_jday <- pheno_out_df$yday_plot[pheno_out$bloomindex]
    zc = max(pheno_out_df$z, na.rm = TRUE)
  }
  
  p_chill <- pheno_out_df %>% 
    ggplot(aes(x = yday_plot)) +
    geom_line(aes(y = y)) +
    geom_hline(yintercept = yc, linetype = 'dashed', col = 'blue') +
    geom_vline(xintercept = bloom_jday, linetype = 'dashed', col = 'black') +
    xlab('Date') +
    ylab('Accumulated Chill\nPortions (y)') +
    scale_x_continuous(breaks = c(275-365, 306-365, 336-365, 1, 32, 61, 92, 122), 
                       labels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May')) +
    coord_cartesian(xlim = xlim,
                    ylim = ylim_chill) +
    theme_bw(base_size = 13) +
    theme(axis.ticks.x = element_blank(),
          axis.title.x = element_blank(),
          axis.text.x = element_blank())
  
  
  p_heat <- pheno_out_df %>% 
    ggplot(aes(x = yday_plot)) +
    geom_line(aes(y = z)) +
    geom_vline(xintercept = bloom_jday, linetype = 'dashed', col = 'black') +
    geom_hline(yintercept = zc, linetype = 'dashed', col = 'red') +
    xlab('Date') +
    ylab('Accumulated\nHeat (z)') +
    scale_x_continuous(breaks = c(275-365, 306-365, 336-365, 1, 32, 61, 92, 122), 
                       labels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May')) +
    coord_cartesian(xlim = xlim, ylim = ylim_heat) +
    theme_bw(base_size = 13)
  
  p_py <- pheno_out_df %>% 
    ggplot(aes(x = yday_plot)) +
    geom_line(aes(y = Py)) +
    geom_vline(xintercept = bloom_jday, linetype = 'dashed', col = 'black') +
    xlab('Date') +
    ylab(expression(
      atop(
        "Share of Effective Heat ("*P[y]*")",
        "(Ontogenetic Competence)"
      ))) +
    scale_x_continuous(breaks = c(275-365, 306-365, 336-365, 1, 32, 61, 92, 122), 
                       labels = c('Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May')) +
    coord_cartesian(xlim = xlim,
                    ylim = ylim_py) +
    theme_bw(base_size = 13) +
    theme(axis.ticks.x = element_blank(),
          axis.title.x = element_blank(),
          axis.text.x = element_blank())
  
  if(annotate_subplot){
    p_temp <- p_temp +
      annotate("label",
               y = Inf,
               x = -Inf, 
               label = "A",
               hjust = -0.2, vjust = 1.2, # slight offset inside plot
               fontface = "bold") 
    
    p_chill <- p_chill +
      annotate("label",
               y = Inf,
               x = -Inf, 
               label = "B",
               hjust = -0.2, vjust = 1.2, # slight offset inside plot
               fontface = "bold") 
    
    p_py <- p_py +
      annotate("label",
               y = Inf,
               x = -Inf, 
               label = "C",
               hjust = -0.2, vjust = 1.2, # slight offset inside plot
               fontface = "bold") 
    
    p_heat <- p_heat +
      annotate("label",
               y = Inf,
               x = -Inf, 
               label = "D",
               hjust = -0.2, vjust = 1.2, # slight offset inside plot
               fontface = "bold") 
  }
  
  
  library(patchwork)
  p_temp / p_chill / p_py / p_heat

}

xlim = c(-90, 122)
ylim_temp = c(-5, 15)
ylim_chill = c(0, 90)
ylim_heat = c(0, 250)
pheno_out = out_normal
yc = 75
s1 = 0.57

viz_processes_phenoflex(pheno_out = out_normal, temp_df = hourtemps[[1]], 
                        yc = yc, s1 = s1, xlim = xlim, ylim_temp = ylim_temp, ylim_chill = ylim_chill, ylim_heat = ylim_heat)

yc = -3.4
s1 = 0.57 
zc = 99.47
ylim_heat = c(0, 250)
ylim_py = c(0, 1)
viz_processes_phenoflex(pheno_out = out, temp_df = hourtemps[[1]], 
                        yc = yc, s1 = s1, xlim = xlim, ylim_temp = ylim_temp, ylim_chill = ylim_chill, ylim_heat = ylim_heat,
                        ylim_py = ylim_py)






