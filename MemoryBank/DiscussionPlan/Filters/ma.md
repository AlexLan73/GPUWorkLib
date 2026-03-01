
# Cкользящие средние 
 - считаем все паралельно по всем лучам, все считаем через размер окна n
https://ru.wikipedia.org/wiki/%D0%A1%D0%BA%D0%BE%D0%BB%D1%8C%D0%B7%D1%8F%D1%89%D0%B0%D1%8F_%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D1%8F%D1%8F

## Простое скользящее среднее SMA
## Экспоненциально взвешенное скользящее среднее EMA
##  Экспоненциальное скользящее среднее произвольного порядка 
  - задавать угол алфа через размер окна = 2/(n+1)
### Экспоненциально взвешенное скользящее среднее MMA  
### Экспоненциально взвешенное скользящее среднее DEMA  
### Экспоненциально взвешенное скользящее среднее TEMA  
   
## Адаптивная скользящая средняя Кауфмана   
https://ru.wikipedia.org/wiki/%D0%90%D0%B4%D0%B0%D0%BF%D1%82%D0%B8%D0%B2%D0%BD%D0%B0%D1%8F_%D1%81%D0%BA%D0%BE%D0%BB%D1%8C%D0%B7%D1%8F%D1%89%D0%B0%D1%8F_%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D1%8F%D1%8F_%D0%9A%D0%B0%D1%83%D1%84%D0%BC%D0%B0%D0%BD%D0%B0

### Адаптивная скользящая средняя AMA
  - со стандартными параметрами 
    - f=2 s=3

## фильтр КАЛМАНА
### зашумленный сигнал 
### 
https://www.google.com/search?q=%D1%84%D0%B8%D0%BB%D1%8C%D1%82%D1%80+%D0%BA%D0%B0%D0%BB%D0%BC%D0%B0%D0%BD%D0%B0+%D0%B4%D0%BB%D1%8F+%D1%84%D0%B8%D0%BB%D1%8C%D1%82%D1%80%D0%B0%D1%86%D0%B8%D0%B8+%D0%B7%D0%B0%D1%88%D1%83%D0%BC%D0%BB%D0%B5%D0%BD%D0%BD%D0%BE%D0%B3%D0%BE+%D0%B2%D1%85%D0%BE%D0%B4%D0%BD%D0%BE%D0%B3%D0%BE+%D1%81%D0%B8%D0%B3%D0%BD%D0%B0%D0%BB%D0%B0&sca_esv=241ff52d71bb1190&biw=1470&bih=799&ei=qIujacGpEM2ri-gPzYjkoQc&ved=2ahUKEwiM0JeZvf2SAxUNzgIHHb7TBe8Q0NsOegQIAxAB&uact=5&sclient=gws-wiz-serp&udm=50&fbs=ADc_l-Y39qeryS7Jwwaqw87B7rGLCqJeM_HZhxbmYJIB16bJpODLB60wYz-FhWYsrvLDMetIHPinBDCua7OG7IhyLHCiQg_aZVRm1mFE3atKWHxGKnMhmF0G1Co8_6ylobLrFQIY52ZROKN7Q3E9aw_NLV9H8gAhcDbDT2L1zCvr8tbW0MHyEjHAlyhMumdUtk_kdl0ZFGAAHfBNoXNWILNYy40JpKfJ3z3xseniG-TeQ62DB8o-cyU6SgmQ5ue8YLsciDa474_4&aep=10&ntc=1&mstk=AUtExfCOOueFd901DQbTkMDOOnRC2I3cd4irL7cnEXnP7x8M3taayV_qyeVnRex7rt_tWerafxugLJ482BD_7YapEuI8vdDpnFuQCkLpO9CcXIJTkvaFsnfyaNkK2rkVSN8Gb_TYtVC7WrC6wckrL44uewaL3W4rLi--wD0&csuir=1


- Стандартный фильтр Калмана предполагает, что система линейна, а шум имеет нормальное (гауссово) распределение. 
- Для нелинейных процессов в радаре (маневрирование объекта) используются модификации, такие как Расширенный фильтр Калмана (EKF) или Сигма-точечный фильтр (UKF). 
 
https://ru.wikipedia.org/wiki/%D0%A4%D0%B8%D0%BB%D1%8C%D1%82%D1%80_%D0%9A%D0%B0%D0%BB%D0%BC%D0%B0%D0%BD%D0%B0

ты знаешь чем я занимаюсь подумай и напиши развернуто могу ли я применять жтот фильтр и как могу применить сделай описание для чайника 


#### Результат 
 - создай два файла с подробным описанием потом их будем использовать как документацию и  поним построим план и таски файлы положи в Doc_Addition\Filters
  - 1 файл все кроме Калмана
  - 2 файл про Калмана