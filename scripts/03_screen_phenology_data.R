adamedor <- read.csv('phenology/adamedor_sub.csv', encoding = 'latin1')

library(tidyverse)

adamedor %>% 
  filter(species == 'Apricot') %>% 
  mutate(Location = location) %>% 
  ggplot(aes(x = cultivar, y = pheno)) + 
  scale_y_continuous(limits = c(30, 100),
                     breaks = c(32,60,91, 121),
                     labels = c('Feb', 'Mar', 'Apr', 'May')) +
  geom_boxplot(aes(fill = Location)) +
  ylab('Full Bloom Observed') +
  xlab('Apricot Cultivar') +
  theme_bw(base_size = 15) +
  theme(legend.position = 'bottom',
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
ggsave('boxplot_apricot.jpeg', height = 20, width = 25, units = 'cm')


adamedor %>% 
  filter(species == 'Sweet Cherry') %>% 
  filter(cultivar %in% c('Burlat', 'Regina', 'Schneiders', 'Sylvia',
                         'Lapins', 'Sam')) %>% 
  mutate(Location = location) %>% 
  ggplot(aes(x = cultivar, y = pheno)) + 
  geom_boxplot(aes(fill = location)) +
  scale_y_continuous(limits = c(75, 130),
                     breaks = c(32,60,91, 121),
                     labels = c('Feb', 'Mar', 'Apr', 'May')) +
  geom_boxplot(aes(fill = Location)) +
  ylab('Full Bloom Observed') +
  xlab('Sweet Cherry Cultivar') +
  theme_bw(base_size = 15) +
  theme(legend.position = 'bottom',
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


adamedor %>% 
  filter(species == 'Almond') %>% 
  filter(cultivar %in% c('Fakhfekh', 'Ferragnes', 'Marcona', 'Tuono',
                         'Malaguena', 'Feraduel', 'Mazzetto', 'Achaak',
                         'Desmayo', 'Garnghzel', 'Genco Taronto')) %>% 
  mutate(Location = location) %>% 
  ggplot(aes(x = cultivar, y = pheno)) + 
  geom_boxplot(aes(fill = location)) +
  scale_y_continuous(limits = c(15, 85),
                     breaks = c(32,60,91, 121),
                     labels = c('Feb', 'Mar', 'Apr', 'May')) +
  geom_boxplot(aes(fill = Location)) +
  ylab('Full Bloom Observed') +
  xlab('Sweet Cherry Cultivar') +
  theme_bw(base_size = 15) +
  theme(legend.position = 'bottom',
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))




adamedor %>% 
  filter(species == 'Almond') %>% 
  filter(location == 'Santomera') %>% 
  #filter(cultivar %in% c('Sylvia', 'Burlat', 'Lapins', 'Schneiders', 'Newstar', 'Lambert', 'Bing', 'Compact Stella')) %>% 
  ggplot(aes(x = cultivar, y = pheno)) + 
  geom_boxplot(aes(fill = location)) +
  theme(legend.position = 'bottom',
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))

adamedor %>% 
  filter(species == 'Sweet Cherry') %>% 
  filter(location == 'Klein-Altendorf') %>% 
  #filter(cultivar %in% c('Sylvia', 'Burlat', 'Lapins', 'Schneiders', 'Newstar', 'Lambert', 'Bing', 'Compact Stella')) %>% 
  ggplot(aes(x = cultivar, y = pheno)) + 
  geom_boxplot(aes(fill = location)) +
  theme(legend.position = 'bottom',
        axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
