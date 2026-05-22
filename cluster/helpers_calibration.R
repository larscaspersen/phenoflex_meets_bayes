#helpers for calibration
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

#evalpheno::custom_PhenoFlex_GDHwrapper
custom_PhenoFlex_GDHwrapper <- function (x, par, constraints = FALSE){
  #x is one of the elements in season List
  #par are the parameters of the model
  
  #make explicit what which parameters is
  yc = par[1]
  zc = par[2]
  s1 = par[3]
  Tu = par[4]
  E0 = par[5]
  E1 = par[6]
  A0 = par[7]
  A1 = par[8]
  Tf = par[9]
  Tc = par[10] 
  Tb = par[11]
  slope = par[12]
  
  
  #in case the parameters do not make sense, return NA
  if(constraints){
    
    t1 <- Tu <= Tb
    t2 <- Tc <= Tb
    t3 <- exp((10 * E0)/(297 * 279)) < 1.5 | exp((10 * E0)/(297 * 279)) > 3.5
    t4 <- exp((10 * E1)/(297 * 279)) < 1.5 | exp((10 * E0)/(297 * 279)) > 3.5
    
    if(any(c(t1, t2, t3, t4))){
      return(NA)
    }
    
  }
  
  #calculat the bloom day
  bloomindex <- chillR::PhenoFlex(temp = x$Temp, 
                                  times = seq_along(x$Temp), 
                                  yc = yc, 
                                  zc = zc, 
                                  s1 = s1, 
                                  Tu = Tu, 
                                  E0 = E0, 
                                  E1 = E1, 
                                  A0 = A0, 
                                  A1 = A1, 
                                  Tf = Tf, 
                                  Tc = Tc, 
                                  Tb = Tb, 
                                  slope = slope, 
                                  Imodel = 0L, 
                                  basic_output = TRUE)$bloomindex
  
  
  
  #return values
  if (bloomindex == 0){
    return(NA)
  } 
  
  JDay <- x$JDay[bloomindex]
  JDaylist <- which(x$JDay == JDay)
  
  #if we are in the norhtern hemisphere and the year corresponding to the index is the smaller one, return negative numbers relative to Jan-1 being 1
  if(length(unique(x$Year)) == 2 & x$Year[bloomindex] == min(x$Year)){
    
    JDay <- JDay - 365
    
  } 
  
  n <- length(JDaylist)
  if (n == 1){
    return(JDay)
  } 
  return(JDay + which(JDaylist == bloomindex)/n - 1/(n/ceiling(n/2)))
  
  
}
