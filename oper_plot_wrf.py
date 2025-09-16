from netCDF4 import Dataset

import numpy as np
from numpy import *

import subprocess

import matplotlib 
matplotlib.use('Agg')  #THIS IS NEEDED IN CHEYENNE
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import json

import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature, ShapelyFeature
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io import shapereader as shpreader

import wrf 

import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy import spatial
from scipy.interpolate import griddata
import scipy.interpolate
import os
import datetime
import time
import shapefile as shp 


starttimeTOTAL = time.time()

arquivos_auxiliares = "arquivos_auxiliares/colors/" 
cmap0 = np.loadtxt(arquivos_auxiliares+"radar_py.rgb", delimiter='	', usecols=(0,1,2), unpack=True)
cmap = np.swapaxes(cmap0,0,1)/255.
cmap_radar = matplotlib.colors.ListedColormap(cmap)
del(cmap0)
del(cmap)
cmap0 = np.loadtxt(arquivos_auxiliares+"wind_py.rgb", delimiter='	', usecols=(0,1,2), unpack=True)
cmap = np.swapaxes(cmap0,0,1)/255.
cmap_wind = matplotlib.colors.ListedColormap(cmap)
del(cmap0)
del(cmap)
cmap0 = np.loadtxt(arquivos_auxiliares+"wind2_py.rgb", delimiter='	', usecols=(0,1,2), unpack=True)
cmap = np.swapaxes(cmap0,0,1)/255.
cmap_wind2 = matplotlib.colors.ListedColormap(cmap)
del(cmap0)
del(cmap)
cmap0 = np.loadtxt(arquivos_auxiliares+"cape_py.rgb", delimiter='	', usecols=(0,1,2), unpack=True)
cmap = np.swapaxes(cmap0,0,1)/255.
cmap_cape = matplotlib.colors.ListedColormap(cmap)
del(cmap0)
del(cmap)
cmap0 = np.loadtxt(arquivos_auxiliares+"thetae_py.rgb", delimiter='	', usecols=(0,1,2), unpack=True)
cmap = np.swapaxes(cmap0,0,1)/255.
cmap_thetae = matplotlib.colors.ListedColormap(cmap)
del(cmap0)
del(cmap)
cmap0 = np.loadtxt(arquivos_auxiliares+"pw_py.rgb", delimiter='	', usecols=(0,1,2), unpack=True)
cmap = np.swapaxes(cmap0,0,1)/255.
cmap_pw = matplotlib.colors.ListedColormap(cmap)
del(cmap0)
del(cmap)
cmap0 = np.loadtxt(arquivos_auxiliares+"IR_py_m90_68.rgb", delimiter='	', usecols=(0,1,2), unpack=True)
cmap = np.swapaxes(cmap0,0,1)/255.
cmap_IR = matplotlib.colors.ListedColormap(cmap)
del(cmap0)
del(cmap)
cmap0 = np.loadtxt(arquivos_auxiliares+"pw_py.rgb", delimiter='	', usecols=(0,1,2), unpack=True)
cmap = np.swapaxes(cmap0,0,1)/255.
cmap_pw = matplotlib.colors.ListedColormap(cmap)
del(cmap0)
del(cmap)

plot1 = True #refl
plot2 = True #mucape
plot3 = True #T2m
plot4 = True #thetae850
plot5 = True # rain total
plot6 = True # wind mag lowest
plot7 = False # precip 1h
plot8 = True # pw


plot_title=True
font_title=12
plot_type = "png"
scale_contour=1
map_line=0.5
font_label=15
flip_barb=True
dx=18000
cp = 1006. #J/kg.K
gamad = 0.01 # 10 K/km

barb_spacing = [10,10,10,10,10]
m=0


today = datetime.datetime.today()
year = str(today.year)
julian_day = str(today.timetuple().tm_yday).zfill(3)

###########################################
#MUDAR AQUI CAMINHO DOS DADOS E DAS FIGURAS
###########################################
dir_data = "/storagefapesp/data/models/wrf/RAW/"+year+"/"+julian_day+"/"
dir_figures = "../figures/" +year+"/"+julian_day+"/"

##################################
##################################

os.makedirs(dir_figures, exist_ok=True)
date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(days=int(julian_day) - 1)
yyyy = date.year
mm = date.month
dd = date.day
hh = '12' 

yyyy_init = str(yyyy)
mm_init = str(mm)
dd_init = str(dd)
hh_init = str(hh) 
yyyymmdd_init = str(yyyy)+str(mm)+str(dd)
hhmm_init = str(hh)+'00'


domain = 'd02'

fcsths = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20', '21','22','23','24','25','26','27','28','29','30','31','32', '33', '34', '35', '36','37','38','39','40','41','42', '43', '44', '45', '46','47','48','49','50','51','52', '53', '54', '55', '56','57','58','59','60','61','62', '63', '64', '65', '66','67','68','69','70','71','72', '73', '74', '75', '76','77','78','79','80','81','82', '83', '84', '85', '86','87','88','89','90','91','92', '93', '94', '95', '96']

crepdecs_feat = ShapelyFeature(shpreader.Reader("./arquivos_auxiliares/crepdecs/CREPDECS_RS.shp").geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='black',linewidth=1.5)

cidades_rs = [
    ("Porto Alegre", -30.0331, -51.2300),
    ("Caxias do Sul", -29.1678, -51.1794),
    ("Pelotas", -31.7719, -52.3426),
    ("Santa Maria", -29.6842, -53.8069),
    ("Gravataí", -29.9444, -50.9919),
    ("Viamão", -30.0819, -51.0233),
    ("Novo Hamburgo", -29.6783, -51.1309),
    ("Rio Grande", -32.0336, -52.0986),
    ("Alvorada", -29.9914, -51.0809),
    ("Passo Fundo", -28.2620, -52.4064),
    ("Sapucaia do Sul", -29.8276, -51.1440),
    ("Uruguaiana", -29.7596, -57.0853),
    ("Santa Cruz do Sul", -29.7184, -52.4250),
    ("Cachoeirinha", -29.9476, -51.0936),
    ("Bagé", -31.3297, -54.1069),
    ("Bento Gonçalves", -29.1662, -51.5165),
    ("Erechim", -27.6366, -52.2697),
    ("Guaíba", -30.1136, -51.3250),
    ("Lajeado", -29.4653, -51.9644),
    ("Santana do Livramento", -30.8900, -55.5328),
    ("Ijuí", -28.3880, -53.9194),
    ("Sapiranga", -29.6380, -51.0069),
    ("Vacaria", -28.5079, -50.9339),
    ("Cruz Alta", -28.6420, -53.6063),
    ("Alegrete", -29.7831, -55.7919),
    ("Venâncio Aires", -29.6143, -52.1932),
    ("Farroupilha", -29.2250, -51.3419),
    ("Cachoeira do Sul", -30.0397, -52.8939),
    ("Santa Rosa", -27.8689, -54.4800),
    ("Santo Ângelo", -28.2990, -54.2663),
    ("Carazinho", -28.2839, -52.7868),
    ("São Gabriel", -30.3341, -54.3219),
    ("Canguçu", -31.3928, -52.6753),
    ("Tramandaí", -29.9847, -50.1324),
    ("Capão da Canoa", -29.7644, -50.0281),
    ("Montenegro", -29.6822, -51.4678),
    ("Taquara", -29.6503, -50.7753),
    ("Parobé", -29.6261, -50.8344),
    ("Osório", -29.8860, -50.2700),
    ("Estância Velha", -29.6530, -51.1739),
    ("Campo Bom", -29.6747, -51.0619),
    ("Eldorado do Sul", -30.0847, -51.6197),
    ("Rosário do Sul", -30.2519, -54.9229),
    ("Dom Pedrito", -30.9828, -54.6717),
    ("Torres", -29.3339, -49.7339),
    ("Panambi", -28.2886, -53.5029),
    ("São Borja", -28.6600, -56.0047),
]
lats_cidades = [lat for (_, lat, lon) in cidades_rs]
lons_cidades = [lon for (_, lat, lon) in cidades_rs]

'''
for nome, lat, lon in cidades_rs:
    ax.text(lon+0.05, lat+0.05, nome, fontsize=7, color="black",transform=mapcrs, zorder=8)
'''

# loop em cada hora de previsão e arquivo do wrf
for fcsth in fcsths:

    
    # gerando horario de cada arquivo para as figuras

    yyyymmddhh_init_as_datetime = datetime.datetime.strptime(yyyymmdd_init+hh_init,"%Y%m%d%H") 
    yyyymmddhh_fcst_as_datetime = yyyymmddhh_init_as_datetime + datetime.timedelta(hours=int(fcsth)) 
    yyyymmdd_hh_fcst = yyyymmddhh_fcst_as_datetime.strftime("%Y%m%d_%H") 
    yyyy_fcst = yyyymmddhh_fcst_as_datetime.strftime("%Y")
    mm_fcst = yyyymmddhh_fcst_as_datetime.strftime("%m")
    dd_fcst = yyyymmddhh_fcst_as_datetime.strftime("%d")
    hh_fcst = yyyymmddhh_fcst_as_datetime.strftime("%H")
    yyyymmdd_fcst = yyyymmddhh_fcst_as_datetime.strftime("%Y%m%d") 
    hhmm_fcst = hh_fcst+'00'

    datahora_str = f"{yyyymmdd_fcst}{str(hhmm_fcst).zfill(4)}"
    datahora_obj = datetime.datetime.strptime(datahora_str, "%Y%m%d%H%M")
    dias_semana = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
    dia_semana = dias_semana[datahora_obj.weekday()]
    
    
    #abrindo o arquivo
    f_name = dir_data+"/wrfout_"+domain+"_"+str(yyyy_fcst)+"-"+str(mm_fcst)+"-"+str(dd_fcst)+"_"+str(hh_fcst)+":00:00" 
    f = Dataset(f_name)

    title = "WRF inicializado "+str(hh_init)+"Z "+str(dd_init)+'/'+str(mm_init)+'/'+str(yyyy_init)     
    titulo_completo = f"Válido: {dia_semana} {yyyymmdd_fcst} {hh_fcst}Z\nHoras de previsão: {fcsth}"

    print("   ")
    print("   ")
    print("________________________________________________________________________________")
    print(title)
    print(titulo_completo)
    print("________________________________________________________________________________")
    print("   ")
    print("   ")
    
    #abrindo as variaveis
    refl = wrf.getvar(f,'REFL_10CM')
    refl_max = np.max(refl, axis=0)
    lats, lons = wrf.latlon_coords(refl_max)
    uh = wrf.getvar(f, 'updraft_helicity', timeidx=0, bottom=2000, top=5000)   # instantaneous uh
    wind10m_max = wrf.getvar(f, 'WSPD10MAX')*3.6   
    

    ter = wrf.getvar(f, 'ter')
    pw = wrf.getvar(f, 'pw')
    slp = wrf.getvar(f, 'slp')
    slp = gaussian_filter(slp, sigma=1.5)
    thetae = wrf.getvar(f, 'theta_e')
    T = wrf.getvar(f, 'tc')
    absvort = wrf.getvar(f, 'avo')/100000
    T2m = wrf.getvar(f, 'T2')-273.15
    mucape,mucin,lcl,lfc = wrf.getvar(f, 'cape_2d')
    mucin = gaussian_filter(mucin, sigma=1.5)
    g = wrf.getvar(f, 'geopt')/98.0665
    z = wrf.getvar(f, 'height')
    zAGL = z - ter
    u = wrf.getvar(f, 'ua', units="m/s")
    u_lowest = u[0,:,:]*1.94384 #kt
    v = wrf.getvar(f, 'va', units="m/s")
    v_lowest = v[0,:,:]*1.94384 #kt
    #w = wrf.getvar(f, 'wa', units="m/s")

    rainc = wrf.getvar(f, 'RAINC')   
    rainnc = wrf.getvar(f, 'RAINNC')
    rainsh = wrf.getvar(f, 'RAINSH') 

    total_precip = rainc + rainnc + rainsh

    mean_lat = np.mean(lats)
    coriolis = 2*0.000072921*sin(mean_lat)
    vort = absvort + coriolis

  
    wind_mag = (u**2. + v**2.)**(1./2.) #m/s
    
    #interpolando algumas variaveis para alturas especificas
    u10m = wrf.getvar(f, 'uvmet10')[0]  #m/s
    v10m = wrf.getvar(f, 'uvmet10')[1]  #m/s
    u1km =  wrf.interplevel(u, zAGL, 1000.).values  #m/s
    v1km =  wrf.interplevel(v, zAGL, 1000.).values  #m/s
    u3km =  wrf.interplevel(u, zAGL, 3000.).values  #m/s
    v3km =  wrf.interplevel(v, zAGL, 3000.).values  #m/s
    u6km =  wrf.interplevel(u, zAGL, 6000.).values  #m/s
    v6km =  wrf.interplevel(v, zAGL, 6000.).values  #m/s
    ushear6km = u6km - u10m
    vshear6km = v6km - v10m
    ushear6km = 1.94384*(u6km - u10m) #kt
    vshear6km = 1.94384*(v6km - v10m) #kt
    u10m = 1.94384*u10m #kt
    v10m = 1.94384*v10m #kt
    u1km = 1.94384*u1km #kt
    v1km = 1.94384*v1km #kt
    u3km = 1.94384*u3km #kt
    v3km = 1.94384*v3km #kt
    u = 1.94384*u #kt
    v = 1.94384*v #kt

    thetae1km =  wrf.interplevel(thetae, zAGL, 1000.).values 
    thetae1km = gaussian_filter(thetae1km, sigma=1)
    vort1km =  wrf.interplevel(vort, zAGL, 1000.).values *10000.
    T3km =  wrf.interplevel(T, zAGL, 3000.).values 
    T3km = gaussian_filter(T3km, sigma=1.5)

    #full
    minlon = np.min(lons)
    maxlon = np.max(lons)
    minlat = np.min(lats)
    maxlat = np.max(lats)

    #tamanho do mapa
    minlon_map = np.min(lons)+0.5
    maxlon_map = np.max(lons)-0.1
    minlat_map = np.min(lats)+0.15
    maxlat_map = np.max(lats)-0.16
   
    projection = 'plate_carree'
    mapcrs = ccrs.PlateCarree()
    datacrs = mapcrs
    
    plot_name = str(int(fcsth))+'.'+plot_type
    
    #plots
    
    #________PLOT 1________________
    if plot1:
     new_dir = dir_figures+"refl/"
     os.makedirs(new_dir, exist_ok=True)
     plotagem = new_dir+plot_name
    
     starttime = time.time()
     plot_n="1"
    
     fig = plt.figure(figsize=(12, 12))
     ax = plt.axes(projection=datacrs)
     ax.set_extent([minlon_map, maxlon_map, minlat_map, maxlat_map], mapcrs)
    
     map_color = 'black'
     ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor=map_color,facecolor='white', linewidth=scale_contour*map_line+1, zorder=1)
     ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line, zorder=2)

     clevs = np.arange(5, 80, 5)
     plot_shaded = ax.contourf(lons, lats, refl_max, clevs, cmap=cmap_radar, transform=mapcrs, zorder=1)
     cbar = plt.colorbar(plot_shaded, pad=0.028, fraction=0.025, aspect=35, orientation='horizontal')
     cbar.ax.tick_params(labelsize=font_label) 
     
     neg_uh_contours = ax.contour(lons, lats, uh, levels=[-100, -50], colors=['black', 'black'], linewidths=[2.5, 2.0], linestyles='solid', transform=mapcrs, zorder=3)
     pos_uh_contours = ax.contour(lons, lats, uh, levels=[30, 60], colors=['grey', 'grey'], linewidths=[2.0, 2.5], linestyles='solid', transform=mapcrs, zorder=3)

     legend_x, legend_y = 0.82, 0.06
     spacing_y = 0.04
     
     box_width, box_height = 0.16, 0.07  # adjust as needed
     background = mpatches.FancyBboxPatch((legend_x - 0.015, legend_y - spacing_y - 0.015),
                                          box_width, box_height,
                                          transform=ax.transAxes,
                                          facecolor='white', edgecolor='none', alpha=0.7,
                                          zorder=2, boxstyle='round,pad=0.01')
     ax.add_patch(background)
     
     ax.add_patch(mpatches.Circle((legend_x, legend_y), 0.015, transform=ax.transAxes,
                                  facecolor='none', edgecolor='black', linewidth=1.5, zorder=3))
     ax.add_patch(mpatches.Circle((legend_x, legend_y), 0.008, transform=ax.transAxes,
                                  facecolor='none', edgecolor='black', linewidth=2.5, zorder=3))
     ax.text(legend_x + 0.02, legend_y, "-50/-100 m²/s²", va='center', ha='left',
             transform=ax.transAxes, fontsize=font_label, zorder=3)
     
     ax.add_patch(mpatches.Circle((legend_x, legend_y - spacing_y), 0.015, transform=ax.transAxes,
                                  facecolor='none', edgecolor='grey', linewidth=1.5, zorder=3))
     ax.add_patch(mpatches.Circle((legend_x, legend_y - spacing_y), 0.008, transform=ax.transAxes,
                                  facecolor='none', edgecolor='grey', linewidth=2.5, zorder=3))
     ax.text(legend_x + 0.02, legend_y - spacing_y, "+30/+60 m²/s²", va='center', ha='left',
             transform=ax.transAxes, fontsize=font_label, zorder=3) 

     ax.scatter(lons_cidades, lats_cidades, s=15, marker="o", facecolors="black", edgecolors="white", transform=mapcrs, zorder=7, label="Cidades")
 

     if plot_title == True:
      plt.title('Refletividade maxima na coluna (dBZ) e helicidade da corrente ascentente 2-5 km (m2/s2)\n'+str(title), loc='left', fontsize=font_title)
      plt.title(titulo_completo, loc='right', fontsize=font_title+1)
     
     plt.tight_layout()
     plt.savefig(plotagem) 
     plt.close(plotagem)
     plt.clf()
     
     cmd = "convert "+plotagem+" -trim "+plotagem
     os.system(cmd)
    
     endtime = time.time()
     print("Tempo para PLOT "+str(plot_n)+": "+str(endtime - starttime)+" segundos")
     print(' ')
    
    #________PLOT 2________________
    if plot2:
     new_dir = dir_figures+"mucape/"
     os.makedirs(new_dir, exist_ok=True)
     plotagem = new_dir+plot_name
     starttime = time.time()
     plot_n="2"
    
     fig = plt.figure(figsize=(12, 12))
     ax = plt.axes(projection=datacrs)
     ax.set_extent([minlon_map, maxlon_map, minlat_map, maxlat_map], mapcrs)
    
     map_color = 'black'
     ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor=map_color,facecolor='white', linewidth=scale_contour*map_line+1, zorder=1)
     ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line, zorder=2)

     clevs = np.arange(250,5250,250)
     plot_shaded = ax.contourf(lons, lats, mucape, clevs, cmap=cmap_cape, zorder=1)
     cbar = plt.colorbar(plot_shaded, ticks=[250, 500, 1000, 1500, 2000, 2500,3000, 3500, 4000, 4500, 5000], pad=0.028, fraction=0.025, aspect=35, orientation='horizontal')
     cbar.ax.tick_params(labelsize=font_label) 

    
     plot_wind = ax.barbs(lons[::barb_spacing[m]+2,::barb_spacing[m]+2], lats[::barb_spacing[m]+2,::barb_spacing[m]+2], ushear6km[::barb_spacing[m]+2,::barb_spacing[m]+2], vshear6km[::barb_spacing[m]+2,::barb_spacing[m]+2], color='black',linewidth=scale_contour*1.3, length=6.5, flip_barb=flip_barb, zorder=4)
    
     clevs_cin_cont = [25,50,100,150]
     plot_contour = ax.contour(lons, lats, mucin, clevs_cin_cont, colors=['green','yellow','red','magenta'], linewidths=scale_contour*2.0, zorder=3)
     plt.clabel(plot_contour, fmt='%d')

     ax.scatter(lons_cidades, lats_cidades, s=15, marker="o", facecolors="black", edgecolors="white", transform=mapcrs, zorder=7, label="Cidades")
    
     if plot_title == True:
      plt.title('CAPE da parcela mais instável (J/kg), cisalhamento 0-6-km\n'+str(title), loc='left', fontsize=font_title)
      plt.title(titulo_completo, loc='right', fontsize=font_title+1)

     plt.tight_layout()
     plt.savefig(plotagem) 
     plt.close(plotagem)
     plt.clf()
     
     cmd = "convert "+plotagem+" -trim "+plotagem
     os.system(cmd)

     endtime = time.time()
     print("Tempo para PLOT "+str(plot_n)+": "+str(endtime - starttime)+" segundos")
     print(' ')
    
    #________PLOT 3________________
    if plot3:
     new_dir = dir_figures+"t2m/"
     os.makedirs(new_dir, exist_ok=True)
     plotagem = new_dir+plot_name
        
     starttime = time.time()
     plot_n="3"
    
     fig = plt.figure(figsize=(12, 12))
     ax = plt.axes(projection=datacrs)
     ax.set_extent([minlon_map, maxlon_map, minlat_map, maxlat_map], mapcrs)
    
     map_color = 'black'
     ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor=map_color,facecolor='white', linewidth=scale_contour*map_line+1, zorder=1)
     ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line, zorder=2)

     clevs_T = np.arange(-40,48,2)
     plot_shaded = ax.contourf(lons, lats, T2m, clevs_T, cmap=cmap_thetae, zorder=1)
     cbar = plt.colorbar(plot_shaded, ticks=[-40, -30, -20, -10, 0, 10, 20, 30, 40], pad=0.028, fraction=0.025, aspect=35, orientation='horizontal')
     cbar.ax.tick_params(labelsize=font_label)
    
     plot_contour1 = ax.contour(lons, lats, T2m, clevs_T, colors='grey',linestyles='-', linewidths=scale_contour*0.8, zorder=3)
     #plt.clabel(plot_contour1, fmt='%d')
    
     plot_contour3 = ax.contour(lons, lats, T2m, [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50], colors='white',linestyles='-', linewidths=scale_contour*1.5, zorder=4)
     plt.clabel(plot_contour3, fmt='%d', fontsize=font_label)
    
     '''
     clevs_slp_cont = np.arange(900,1100,2)
     slp_color = 'black'
     plot_contour2 = ax.contour(lons, lats, slp, clevs_slp_cont, colors='black', linewidths=scale_contour*3.0, zorder=5)
     plt.clabel(plot_contour2, fmt='%d', fontsize=font_label)
     '''
    
     plot_wind = ax.barbs(lons[::barb_spacing[m],::barb_spacing[m]], lats[::barb_spacing[m],::barb_spacing[m]], u10m[::barb_spacing[m],::barb_spacing[m]], v10m[::barb_spacing[m],::barb_spacing[m]], pivot='middle', color='black',linewidth=scale_contour*1.3, length=6.5, flip_barb=flip_barb, zorder=6)
     

     ax.scatter(lons_cidades, lats_cidades, s=15, marker="o", facecolors="black", edgecolors="white", transform=mapcrs, zorder=7, label="Cidades")
    
     if plot_title == True:
      plt.title('Temperatura em 2m (C), vento em 10 m, pressão ao nível médio do mar (hPa)\n'+str(title), loc='left', fontsize=font_title)
      plt.title(titulo_completo, loc='right', fontsize=font_title+1)
     
     plt.tight_layout()
     plt.savefig(plotagem) 
     plt.close(plotagem)
     plt.clf()
     
    
     cmd = "convert "+plotagem+" -trim "+plotagem
     os.system(cmd)

     endtime = time.time()
     print("Tempo para PLOT "+str(plot_n)+": "+str(endtime - starttime)+" segundos")
     print(' ')

    #________PLOT 4________________
    if plot4:

     new_dir = dir_figures+"vort_1km/"
     os.makedirs(new_dir, exist_ok=True)
     plotagem = new_dir+plot_name
        
     starttime = time.time()
     plot_n="4"
    
     fig = plt.figure(figsize=(12, 12))
     ax = plt.axes(projection=datacrs)
     ax.set_extent([minlon_map, maxlon_map, minlat_map, maxlat_map], mapcrs)
    
     map_color = 'black'
     ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor=map_color,facecolor='white', linewidth=scale_contour*map_line+1, zorder=1)
     ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line, zorder=2)

     clevs_vort = np.arange(-110,0,10)
     plot_shaded = ax.contourf(lons, lats, vort1km, clevs_vort, cmap=plt.cm.cividis, zorder=1)
     cbar = plt.colorbar(plot_shaded, pad=0.028, fraction=0.025, aspect=35, orientation='horizontal')
     cbar.ax.tick_params(labelsize=font_label)
    
     plot_contour2 = ax.contour(lons, lats, refl_max, [50], colors='blue', linewidths=scale_contour*1.5, zorder=5)

    
     plot_wind = ax.barbs(lons[::barb_spacing[m],::barb_spacing[m]], lats[::barb_spacing[m],::barb_spacing[m]], u1km[::barb_spacing[m],::barb_spacing[m]], v1km[::barb_spacing[m],::barb_spacing[m]], pivot='middle', color='black',linewidth=scale_contour*1.3, length=6.5, flip_barb=flip_barb, zorder=6)
     
     ax.scatter(lons_cidades, lats_cidades, s=15, marker="o", facecolors="black", edgecolors="white", transform=mapcrs, zorder=7, label="Cidades")

    
     if plot_title == True:
      plt.title('Vorticidade (10^-4 s^-1) e vento em 1 km, refletividade 50 dBZ (azul)\n'+str(title), loc='left', fontsize=font_title)
      plt.title(titulo_completo, loc='right', fontsize=font_title+1)
     
     plt.tight_layout()
     plt.savefig(plotagem) 
     plt.close(plotagem)
     plt.clf()
     
     cmd = "convert "+plotagem+" -trim "+plotagem
     os.system(cmd)

     endtime = time.time()
     print("Tempo para PLOT "+str(plot_n)+": "+str(endtime - starttime)+" segundos")
     print(' ')

    
    #________PLOT 5________________
    if plot5:
     new_dir = dir_figures+"total_precip/"
     os.makedirs(new_dir, exist_ok=True)
     plotagem = new_dir+plot_name
     starttime = time.time()
     plot_n="5"
    
     fig = plt.figure(figsize=(12, 12))
     ax = plt.axes(projection=datacrs)
     ax.set_extent([minlon_map, maxlon_map, minlat_map, maxlat_map], mapcrs)
    
     map_color = 'black'
     ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor=map_color,facecolor='white', linewidth=scale_contour*map_line+1, zorder=1)
     ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     #ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line, zorder=2)

     with open(arquivos_auxiliares+"colormap_qpe.json") as f:
         colormap = json.load(f)
     original_data = colormap["data"]
     color_0mm = None
     filtered_data = []
     for level, color in original_data:
         if level == 0:
             color_0mm = color
         else:
             filtered_data.append([level, color])  
     new_data = [[0, [255, 255, 255, 255]], [1, color_0mm]] + filtered_data
     clevs = [entry[0] for entry in new_data]
     rgba_colors = np.array([entry[1] for entry in new_data], dtype=np.float32) / 255.0 
     cmap_precip = ListedColormap(rgba_colors)
     norm_precip = BoundaryNorm(clevs, cmap_precip.N)

     plot_shaded = ax.contourf(lons, lats, total_precip, levels=clevs, cmap=cmap_precip, norm=norm_precip, transform=mapcrs, zorder=1)

     cbar = plt.colorbar(plot_shaded, pad=0.028, fraction=0.025, aspect=35, orientation='horizontal')
     cbar.ax.tick_params(labelsize=font_label) 

     ax.add_feature(crepdecs_feat, zorder=4)

     ax.scatter(lons_cidades, lats_cidades, s=15, marker="o", facecolors="black", edgecolors="white", transform=mapcrs, zorder=7, label="Cidades")
    
     if plot_title == True:
      plt.title('Precipitação total (mm)\n'+str(title), loc='left', fontsize=font_title)
      plt.title(titulo_completo, loc='right', fontsize=font_title+1)
     
     plt.tight_layout()
     plt.savefig(plotagem) 
     plt.close(plotagem)
     plt.clf()
     
     cmd = "convert "+plotagem+" -trim "+plotagem
     os.system(cmd)
    
     endtime = time.time()
     print("Tempo para PLOT "+str(plot_n)+": "+str(endtime - starttime)+" segundos")
     print(' ')

    #________PLOT 6________________
    if plot6:
     new_dir = dir_figures+"wind_vel/"
     os.makedirs(new_dir, exist_ok=True)
     plotagem = new_dir+plot_name
     starttime = time.time()
     plot_n="6"
    
     fig = plt.figure(figsize=(12, 12))
     ax = plt.axes(projection=datacrs)
     ax.set_extent([minlon_map, maxlon_map, minlat_map, maxlat_map], mapcrs)
    
     map_color = 'black'
     ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor=map_color,facecolor='white', linewidth=scale_contour*map_line+1, zorder=1)
     ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line, zorder=2)

     clevs_wind = np.arange(20,120,5)
     plot_shaded = ax.contourf(lons, lats, wind10m_max, levels=clevs_wind, cmap=cmap_wind, transform=mapcrs, zorder=1)
     cbar = plt.colorbar(plot_shaded, pad=0.028, fraction=0.025, aspect=35, orientation='horizontal')
     cbar.ax.tick_params(labelsize=font_label) 

     refl_contours = ax.contour(lons, lats, refl_max, levels=[50], colors=['blue'], linewidths=[1.7], linestyles='solid', transform=mapcrs, zorder=3)

     plot_wind = ax.barbs(lons[::barb_spacing[m],::barb_spacing[m]], lats[::barb_spacing[m],::barb_spacing[m]], u10m[::barb_spacing[m],::barb_spacing[m]], v10m[::barb_spacing[m],::barb_spacing[m]], pivot='middle', color='black',linewidth=scale_contour*1.3, length=6.5, flip_barb=flip_barb, zorder=6)

     ax.scatter(lons_cidades, lats_cidades, s=15, marker="o", facecolors="black", edgecolors="white", transform=mapcrs, zorder=7, label="Cidades")

    
     if plot_title == True:
      plt.title('Velocidade do vento em 10 m máxima na última hora (km/h) e refletividade de 50 dBZ (azul)\n'+str(title), loc='left', fontsize=font_title)
      plt.title(titulo_completo, loc='right', fontsize=font_title+1)
     
     plt.tight_layout()
     plt.savefig(plotagem) 
     plt.close(plotagem)
     plt.clf()
     
     cmd = "convert "+plotagem+" -trim "+plotagem
     os.system(cmd)
    
     endtime = time.time()
     print("Tempo para PLOT "+str(plot_n)+": "+str(endtime - starttime)+" segundos")
     print(' ')


    #________PLOT 7________________
    if plot7:
     new_dir = dir_figures+"precip_1h/"
     os.makedirs(new_dir, exist_ok=True)
     plotagem = new_dir+plot_name
     starttime = time.time()
     plot_n="7"
    
     fig = plt.figure(figsize=(12, 12))
     ax = plt.axes(projection=datacrs)
     ax.set_extent([minlon_map, maxlon_map, minlat_map, maxlat_map], mapcrs)
    
     map_color = 'black'
     ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor=map_color,facecolor='white', linewidth=scale_contour*map_line+1, zorder=1)
     ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line, zorder=2)

     with open(arquivos_auxiliares+"colormap_qpe.json") as f:
         colormap = json.load(f)
     original_data = colormap["data"]
     color_0mm = None
     filtered_data = []
     for level, color in original_data:
         if level == 0:
             color_0mm = color
         else:
             filtered_data.append([level, color])  
     new_data = [[0, [255, 255, 255, 255]], [1, color_0mm]] + filtered_data
     clevs = [entry[0] for entry in new_data]
     rgba_colors = np.array([entry[1] for entry in new_data], dtype=np.float32) / 255.0 
     cmap_precip = ListedColormap(rgba_colors)
     norm_precip = BoundaryNorm(clevs, cmap_precip.N)

     plot_shaded = ax.contourf(lons_extremes, lats_extremes, precip_1h, levels=clevs, cmap=cmap_precip, norm=norm_precip, transform=mapcrs, zorder=1)

     cbar = plt.colorbar(plot_shaded, pad=0.028, fraction=0.025, aspect=35, orientation='horizontal')
     cbar.ax.tick_params(labelsize=font_label) 

     ax.add_feature(crepdecs_feat, zorder=4)

     ax.scatter(lons_cidades, lats_cidades, s=15, marker="o", facecolors="black", edgecolors="white", transform=mapcrs, zorder=7, label="Cidades")
    
     if plot_title == True:
      plt.title('Precipitação na última hora (mm)\n'+str(title), loc='left', fontsize=font_title)
      plt.title(titulo_completo, loc='right', fontsize=font_title+1)
     
     plt.tight_layout()
     plt.savefig(plotagem) 
     plt.close(plotagem)
     plt.clf()
     
     cmd = "convert "+plotagem+" -trim "+plotagem
     os.system(cmd)
    
     endtime = time.time()
     print("Tempo para PLOT "+str(plot_n)+": "+str(endtime - starttime)+" segundos")
     print(' ')

    #________PLOT 8________________
    if plot8:
     new_dir = dir_figures+"pw_3km/"
     os.makedirs(new_dir, exist_ok=True)
     plotagem = new_dir+plot_name
     starttime = time.time()
     plot_n="8"
    
     fig = plt.figure(figsize=(12, 12))
     ax = plt.axes(projection=datacrs)
     ax.set_extent([minlon_map, maxlon_map, minlat_map, maxlat_map], mapcrs)
    
     map_color = 'black'
     ax.add_feature(cfeature.LAND.with_scale('10m'), edgecolor=map_color,facecolor='white', linewidth=scale_contour*map_line+1, zorder=1)
     ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line+1, zorder=2)
     ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=map_color, linewidth=scale_contour*map_line, zorder=2)

     clevs_pw = np.arange(20,72,2)
     plot_shaded = ax.contourf(lons, lats, pw, clevs_pw, cmap=cmap_pw, zorder=1)
     cbar = plt.colorbar(plot_shaded, ticks=[20,30,40,50,60,70], pad=0.028, fraction=0.025, aspect=35, orientation='horizontal')
     cbar.ax.tick_params(labelsize=font_label)

     plot_contour = ax.contour(lons, lats, T3km, levels=np.arange(-30,30,2), colors=['red'], linewidths=[2.2], linestyles='--', transform=mapcrs, zorder=3)
     plt.clabel(plot_contour, fmt='%d', fontsize=font_label)
    

     plot_wind = ax.barbs(lons[::barb_spacing[m],::barb_spacing[m]], lats[::barb_spacing[m],::barb_spacing[m]], u3km[::barb_spacing[m],::barb_spacing[m]], v3km[::barb_spacing[m],::barb_spacing[m]], pivot='middle', color='black',linewidth=scale_contour*1.3, length=6.5, flip_barb=flip_barb, zorder=6)
     
     ax.scatter(lons_cidades, lats_cidades, s=15, marker="o", facecolors="black", edgecolors="white", transform=mapcrs, zorder=7, label="Cidades")
    
     if plot_title == True:
      plt.title('Água precipitável (mm), temperatura e vento em 3 km\n'+str(title), loc='left', fontsize=font_title)
      plt.title(titulo_completo, loc='right', fontsize=font_title+1)
     
     plt.tight_layout()
     plt.savefig(plotagem) 
     plt.close(plotagem)
     plt.clf()
     
     cmd = "convert "+plotagem+" -trim "+plotagem
     os.system(cmd)

     endtime = time.time()
     print("Tempo para PLOT "+str(plot_n)+": "+str(endtime - starttime)+" segundos")
     print(' ')
    


    plt.close('all')
