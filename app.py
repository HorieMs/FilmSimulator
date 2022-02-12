import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tmm
import colour
import os
import datetime


wl_min=400.0
wl_max=800.0
wl_pitch=1.0

n_env=1.0
inc_angle=0.0
nlayers=1

nk_idx_subst=0
nk_idx_film=0

def order_n(i): return {1:"1st", 2:"2nd", 3:"3rd"}.get(i) or "%dth"%i

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

# @st.cache
def get_nk_list():
    """
    フォルダ内のnkファイル名一覧の取得
    Parameters
    ----------
    nk_path : str
        nkファイルのパス.
    Returns
    -------
    name_list : list of str
        ファイル名のリスト
        
    """
    
    nk_list=[]
    # nk_fpath=".\\data\\nk\\"
    # nk_dirs=nk_fpath+"*.nk"
    #nk_dirs="data\\nk\\*.nk"
    #nk_files=glob.glob(nk_dirs)
    
    cwd=os.getcwd()
    st.write(cwd)


    files=os.listdir(".")
    st.write(files)
    files=os.listdir("data")
    st.write(files)


    
    nk_dirs="data\\nk"
    files=os.listdir(nk_dirs)
    nk_files=[f for f in files if os.path.isfile(os.path.join(nk_dirs, f))]
 
    for nk_file in nk_files:
        basename = os.path.splitext(os.path.basename(nk_file))[0]
        nk_list.append(basename)    
    
    if len(nk_list)<1:
        st.error('not find nk data in '+nk_dirs)
        files_data=glob.glob("data")
        st.error('dir of data =',files_data)
        files_nk=glob.glob("data\\nk")
        st.error('dir of data_nk =',files_nk)

    
    return nk_list

def calc_nk_list(nk_fn_list,wl):
    """
    各層の光学定数の関数リストと与えられた波長から、薄膜の光学定数リストを返す

    Parameters
    ----------
    nk_fn_list : list of fn(wl)
        光学定数の関数リスト.
    wl : float
        波長(nm).

    Returns
    -------
    nk_list : array of complex
        各層の工学定数.

    """
    
    nk_list=[]
    for nk in nk_fn_list:
        nk_list.append(nk(wl))
    return nk_list


# @st.cache
def make_nk_fn(nk_name_list=[]):
    """
    各層の光学定数の関数を返す
    Parameters
    ----------
    nk_name_list : list of string
        光学定数名のリスト.

    Returns
    -------
    nk_fn_list : list of fn(wl)
        各層の光学定数の関数リスト.

    """
    nk_path='.\\data\\nk\\' # nkファイルのパス
    nk_fn_list=[]
    for idx,nk_name in enumerate(nk_name_list):
        if isinstance(nk_name,complex) or isinstance(nk_name,float) or isinstance(nk_name,int):
            nk=nk_name
            nk_fn = lambda wavelength: nk
        elif isinstance(nk_name,str) and str(nk_name).isnumeric():
            nk=float(nk_name)
            nk_fn = lambda wavelength: nk
        else:
            fname_path=nk_path+nk_name+'.nk'
            if os.path.isfile(fname_path):
                nk_mat=np.loadtxt(fname_path,comments=';',encoding="utf-8_sig")
                w_mat=nk_mat[:,0]
                n_mat=np.array(nk_mat[:,1]+nk_mat[:,2]*1j)    
                nk_fn= interp1d(w_mat,n_mat, kind='quadratic')
            else:
                try:
                    nk=complex(nk_name)
                except ValueError:
                    nk=complex(1.0)
                nk_fn = lambda wavelength: nk
        
        nk_fn_list.append(nk_fn)
    
    return nk_fn_list

def calc_reflectance(wl_ar,nk_fn_list,d_list,inc_angle=0.0):
    """
    光学薄膜の反射率を計算
    Parameters
    ----------
    wl_ar : array of float
        波長(nm).
    nk_fn_list : list of fn(wl)
        光学定数関数のリスト.
    d_list : array pof float
        各層の膜厚(nm).
        単層膜: [np.inf, 300, np.inf]  # 最初の要素が媒質
    inc_angle : float, optional
        入射角(deg). The default is 0.0.

    Returns
    -------
    Rp_ar : array of float
        Rp.
    Rs_ar : array of float
        Rs.

    """
    
    Rp_ar=np.empty(len(wl_ar),dtype=float)
    Rs_ar=np.empty(len(wl_ar),dtype=float)
    inc_angle_rad=inc_angle/180.0*np.pi

    for idx,wl in enumerate(wl_ar):
        n_list=calc_nk_list(nk_fn_list,wl) 
        Rp_ar[idx]=tmm.coh_tmm('p', n_list, d_list, inc_angle_rad, wl)['R']
        if inc_angle<0.01:
            Rs_ar[idx]=Rp_ar[idx]
        else:
            Rs_ar[idx]=tmm.coh_tmm('s', n_list, d_list, inc_angle_rad, wl)['R']

    return (Rp_ar,Rs_ar)



st.title('Optical film simulator')

st.sidebar.header('Light parameters')




inc_angle=st.sidebar.number_input('Incident angle [deg]',min_value=0.0,max_value=89.0,value=0.0,step=0.1,format='%3.1f')
wl_option=st.sidebar.selectbox('Wavelength(nm)',('Visible','General'))
if wl_option=='Visible':
    wl_min=380.0
    wl_max=780.0
    wl_pitch=5.0

wl_range=st.sidebar.slider('Wavelength range [nm]',min_value=200.0,max_value=1000.0,value=(wl_min,wl_max),step=1.0,format='%.0f')
if wl_range:
    wl_min=wl_range[0]
    wl_max=wl_range[1]


value=st.sidebar.number_input('Wavelength pitch [nm]',min_value=0.1,max_value=10.0,value=wl_pitch,step=0.1,format='%3.1f')
if value:
    wl_pitch=value

st.sidebar.header('Atmosphere')
value=st.sidebar.number_input('Refractive index (air:1.00)',min_value=1.0,max_value=3.0,value=1.0,step=0.01,format='%3.2f')
if value:
    n_env=value


st.header('Film stack')
nlayers=st.number_input('Number of layer',min_value=1,max_value=5,value=nlayers,step=1,format='%d')

nk_namelist=get_nk_list()
if len(nk_namelist)<1:
    st.error('nk list not find')

nk_idx_subst=nk_namelist.index('Silicon')
nk_idx_film=nk_namelist.index('SiO2')
# print('nk list',nk_namelist)

nk_name_list=[]
d_list=[]
nk_name_list.append(1)
d_list.append(np.Inf)

for num in range(nlayers):
    col1,col2=st.columns((2,1))
    label_layer=order_n(num+1)+' layer'
    with col1:
        nk_name=st.selectbox(label_layer,nk_namelist,index=nk_idx_film-3*num,key='L'+str(num+1))
        nk_name_list.append(nk_name)
    with col2:
        val=st.number_input('thickness[nm]',min_value=0.1,max_value=1e6,value=100.0,step=0.1,format='%g',key='T'+str(num+1))
        d_list.append(val)

nk_name=st.selectbox('substrate',nk_namelist,index=nk_idx_subst,key='L0')
nk_name_list.append(nk_name)
d_list.append(np.Inf)


nk_fn_list=make_nk_fn(nk_name_list)

wl_ar=np.arange(wl_min,wl_max+wl_pitch,wl_pitch,dtype=float)
Rp=np.zeros(len(wl_ar),dtype=float)
Rs=np.zeros(len(wl_ar),dtype=float)

Rp,Rs=calc_reflectance(wl_ar,nk_fn_list,d_list,inc_angle)


fig=plt.figure()
if inc_angle<0.01:
    plt.plot(wl_ar, Rp, 'green')
    title_msg=f'Reflection at {round(inc_angle,1)}$^\circ$ incidence, R(green)'
    plt.legend(['R(nominal)'])
else:
    plt.plot(wl_ar, Rp, 'red', wl_ar, Rs, 'blue',wl_ar,(Rp+Rs)/2.0,'green')
    title_msg=f'Reflection at {round(inc_angle,1)}$^\circ$ incidence, Rp(red),Rs,(blue),Rn(green)'
    plt.legend(['Rp','Rs','R(mean)'])    

plt.xlabel('wavelength(nm)')
plt.ylabel('Reflectance')

plt.title(title_msg)
st.pyplot(fig)




colour_wl_min=380
colour_wl_max=780
colour_wl_pitch=5
colour_wl_ar=np.arange(colour_wl_min,colour_wl_max+colour_wl_pitch,wl_pitch,dtype=float)
if len(wl_ar)!=len(colour_wl_ar) or any(wl_ar!=colour_wl_ar):
    colour_wl_ar=np.arange(colour_wl_min,colour_wl_max+colour_wl_pitch,colour_wl_pitch,dtype=float)
    colour_Rp=np.zeros(len(wl_ar),dtype=float)
    colour_Rs=np.zeros(len(wl_ar),dtype=float)
    colour_Rp,colour_Rs=calc_reflectance(colour_wl_ar,nk_fn_list,d_list,inc_angle)
else:
    colour_Rp,colour_Rs,colour_wl_ar=Rp,Rs,wl_ar


sd_p = colour.SpectralDistribution(colour_Rp, name='Sample Rp')    
sd_s = colour.SpectralDistribution(colour_Rs, name='Sample Rs')
sd_p.wavelengths=np.arange(colour_wl_min,colour_wl_max+colour_wl_pitch,colour_wl_pitch)
sd_s.wavelengths=np.arange(colour_wl_min,colour_wl_max+colour_wl_pitch,colour_wl_pitch)



# Convert to Tristimulus Values
cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
illuminant = colour.ILLUMINANTS_SDS['D65']

# Calculating the sample spectral distribution *CIE XYZ* tristimulus values.
XYZ_p = colour.sd_to_XYZ(sd_p, cmfs, illuminant)
XYZ_s = colour.sd_to_XYZ(sd_s, cmfs, illuminant)
RGB_p = colour.XYZ_to_sRGB(XYZ_p / 100)
RGB_s = colour.XYZ_to_sRGB(XYZ_s / 100)
b_p=[]
for v in RGB_p:
    b_p.append(np.clip(round(v*255),0,255))
b_s=[]
for v in RGB_s:
    b_s.append(np.clip(round(v*255),0,255))

strRGB_p='#'+format(b_p[0], 'x')+format(b_p[1], 'x')+format(b_p[2], 'x')
strRGB_s='#'+format(b_s[0], 'x')+format(b_s[1], 'x')+format(b_s[2], 'x')

col1,col2=st.columns(2)
with col1:
    color_p = st.color_picker('Film colour (Rp)', strRGB_p,key='cp_Rp')
    st.write('XYZ chromaticity',XYZ_p)
with col2:
    color_s = st.color_picker('Film colour (Rs)', strRGB_s,key='cp_Rs')
    st.write('XYZ chromaticity',XYZ_s)



nwl=len(wl_ar)
if inc_angle<0.01:
    data=np.concatenate([wl_ar.reshape([nwl,1]),Rp.reshape([nwl,1])],1)
    data_x=np.array([-1,-2,-3,-11,-12,-13]).reshape(6,1)
    data_y=np.array([XYZ_p[0],XYZ_p[1],XYZ_p[2],RGB_p[0],RGB_p[1],RGB_p[2]]).reshape(6,1)
    data_xy=np.concatenate([data_x,data_y],1)
    data_all=np.concatenate([data_xy,data],0)
else:
    data=np.concatenate([wl_ar.reshape([nwl,1]),Rp.reshape([nwl,1]),Rs.reshape([nwl,1])],1)
    data_x=np.array([-1,-2,-3,-11,-12,-13]).reshape(6,1)
    data_y=np.array([XYZ_p[0],XYZ_p[1],XYZ_p[2],RGB_p[0],RGB_p[1],RGB_p[2]]).reshape(6,1)
    data_z=np.array([XYZ_p[0],XYZ_s[1],XYZ_s[2],RGB_s[0],RGB_s[1],RGB_s[2]]).reshape(6,1)
    data_xyz=np.concatenate([data_x,data_y,data_z],1)
    data_all=np.concatenate([data_xyz,data],0)

df=pd.DataFrame(data_all)
if inc_angle<0.01:
    df.columns = ['Wavelength(nm)', 'R']
else:
    df.columns = ['Wavelength(nm)', 'Rp', 'Rs']

df.set_index('Wavelength(nm)')

#st.write(df)
#np.savetxt(".\\data\\temp\\data.csv",data,fmt='%.5f',delimiter=',') 

csv = convert_df(df)

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
# YYYYMMDDhhmmss形式に書式化
d = now.strftime('%Y%m%d%H%M%S')
fname='data_'+d+'.csv'

st.download_button(
     label="Download data as CSV",
     data=csv,
     file_name=fname,
     mime='text/csv',
 )
