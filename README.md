# wrf-figures-svr
O script `oper_plot_wrf.py` gera 7 figuras para cada saída do WRF. Por padrão, os arquivos `wrfout*` do domínio `d01` são utilizados, a data atual é empregada para localizar os dados e as figuras são salvas no diretório `figures/ano/dia_juliano` relativo ao repositório.

Esses valores padrão podem ser substituídos pela interface de linha de comando:

```
python oper_plot_wrf.py --dominio d02 --output-dir /caminho/para/figuras --raw-dir /caminho/para/wrf
```

O script também usa imagemagick para cortar as bordas brancas das figuras.

Os arquivos no diretório `arquivos_auxiliares` contêm escalas de cores e shapefiles que são usados em algumas figuras.

O ambiente python utilizado nas plotagens está no arquivo `environment.yml`.
