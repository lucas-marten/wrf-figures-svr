# wrf-figures-svr
O script oper_plot_wrf.py gera 7 figuras para cada dado do WRF. O script usa os arquivos wrfout* do domínio d02 apenas. O script utiliza o ano e dia juliano corrente para acessar o diretório onde estão os dados do WRF, e gera figuras no diretório ano/dia_juliano/. 
***O caminho dos dados e das figuras deve ser ajustado nas linhas 114-115***

O script também usa imagemagick para cortar as bordas brancas das figuras.

Os arquivos no diretório arquivos_auxiliares contêm escalas de cores e shapefiles que são usados em algumas figuras.

O ambiente python utilizado nas plotagens está no arquivo environment.yml
