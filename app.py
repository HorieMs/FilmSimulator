import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import tmm
import colour
import os
import time
import datetime
import glob


st.set_page_config(
     page_title="Optical film simulator",
     page_icon="🌈",
     layout="wide",
     initial_sidebar_state="auto",
     menu_items={
         'Get Help': 'https://koumyou.org/optical-film-simulator/',
         'Report a bug': "https://koumyou.org/optical-film-simulator/",
         'About': "# This is an optical film sumulator app!"
     }
 )

calc_mode_menu=['Wavelength Scan','Incident angle Scan']
calc_mode='Wavelength Scan'

wl_min=400.0
wl_max=800.0
wl_pitch=5.0
inc_angle=0.0

inc_wl=600.0
inc_angle_min=0.0
inc_angle_max=70.0
inc_angle_pitch=1.0


n_env=1.0
nlayers=1

nk_idx_subst=0
nk_idx_film=0


def tictoc(func):
    def _wrapper(*args,**keywargs):
        start_time=time.time()
        result=func(*args,**keywargs)
        print('time: {:.9f} [sec]'.format(time.time()-start_time))
        return result
    return _wrapper



def order_n(i): return {1:"1st (top)", 2:"2nd", 3:"3rd"}.get(i) or "%dth"%i

@st.cache_data
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

@st.cache_data
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
    nk_dirs="data//nk"
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

    nk_list.sort()
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
        各層の光学定数.

    """
    
    nk_list=[]
    for nk in nk_fn_list:
        nk_list.append(nk(wl))
    # nk_list=[]
    # n=len(nk_fn_list)
    # for k in range(n):
    #     val=nk_fn_list[k](wl)
    #     nk_list.append(val)
    #print(nk_list)
    return nk_list



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
    #nk_dirs="data//nk"
    nk_path="data//nk//" # nkファイルのパス
    nk_fn_list=[]
    for idx,nk_name in enumerate(nk_name_list):
        if isinstance(nk_name,complex) or isinstance(nk_name,float) or isinstance(nk_name,int):
            nk=complex(nk_name)
            nk_fn = lambda wavelength: nk
            #print(f'Idx={idx},Instance==numeric, val={nk}')
        elif isinstance(nk_name,str) and str(nk_name).isnumeric():
            nk=float(nk_name)
            nk_fn = lambda wavelength: nk
            #print(f'Idx={idx},Instance==str, val={nk}')
        else:
            fname_path=nk_path+nk_name+'.nk'
            if os.path.isfile(fname_path):
                nk_mat=np.loadtxt(fname_path,comments=';',encoding="utf-8_sig")
                #st.write(nk_mat)
                w_mat=nk_mat[:,0]
                n_mat=np.array(nk_mat[:,1]+nk_mat[:,2]*1j)    
                #nk_fn= interp1d(w_mat,n_mat, kind='quadratic', fill_value='extrapolate')
                nk_fn= interp1d(w_mat,n_mat, kind='linear', fill_value='extrapolate')
                #print(f'Idx={idx},Instance=={fname_path} exist')
            else:
                try:
                    nk=complex(nk_name)
                except ValueError:
                    nk=complex(1.0)
                #print(f'Idx={idx},Instance=={fname_path} not exist, nk={nk}')
                nk_fn = lambda wavelength: nk
        
        nk_fn_list.append(nk_fn)
    
    #print(nk_fn_list)
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
    inc_angle_rad=float(inc_angle/180.0*np.pi)

    for idx,wl in enumerate(wl_ar):
        n_list=calc_nk_list(nk_fn_list,float(wl)) 
        #print(f"{wl}nm: n={n_list}")
        Rp_ar[idx]=tmm.coh_tmm('p', n_list, d_list, inc_angle_rad, float(wl))['R']
        if inc_angle<0.01:
            Rs_ar[idx]=Rp_ar[idx]
        else:
            Rs_ar[idx]=tmm.coh_tmm('s', n_list, d_list, inc_angle_rad, wl)['R']
            
    return (Rp_ar,Rs_ar)


def calc_angle_reflectance(wl,nk_fn_list,d_list,angle_ar):
    """
    光学薄膜の反射率を計算
    Parameters
    ----------
    wl : 波長(nm).
    nk_fn_list : list of fn(wl)
        光学定数関数のリスト.
    d_list : array of float
        各層の膜厚(nm).
        単層膜: [np.inf, 300, np.inf]  # 最初の要素が媒質
    angle_ar : array of float, 入射角の配列

    Returns
    -------
    Rp_ar : array of float
        Rp.
    Rs_ar : array of float
        Rs.

    """
    
    Rp_ar=np.empty(len(angle_ar),dtype=float)
    Rs_ar=np.empty(len(angle_ar),dtype=float)
    n_list=calc_nk_list(nk_fn_list,float(wl))
    #print(f"{wl}nm: n={n_list}")
    for idx,inc_angle in enumerate(angle_ar):
        inc_angle_rad=inc_angle/180.0*np.pi
        Rp_ar[idx]=tmm.coh_tmm('p', n_list, d_list, inc_angle_rad, wl)['R']
        if inc_angle<0.01:
            Rs_ar[idx]=Rp_ar[idx]
        else:
            Rs_ar[idx]=tmm.coh_tmm('s', n_list, d_list, inc_angle_rad, wl)['R']

    return (Rp_ar,Rs_ar)


def disp_wavelength_scan():
    """
    
    """
    wl_ar=np.arange(wl_min,wl_max+wl_pitch,wl_pitch,dtype=float)
    Rp=np.zeros(len(wl_ar),dtype=float)
    Rs=np.zeros(len(wl_ar),dtype=float)

    Rp,Rs=calc_reflectance(wl_ar,nk_fn_list,d_list,inc_angle)

    st.subheader('Spectrum')


    fig = go.Figure()

    if inc_angle<0.01:
        fig.add_trace(go.Scatter(
            x=wl_ar, y=Rp,
            name='R(nominal)',
            mode='lines',
            marker_color='rgba(0, 0, 0, .8)'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=wl_ar, y=Rp,
            name='Rp',
            mode='lines',
            marker_color='rgba(255, 0, 0, .8)'
        ))
        fig.add_trace(go.Scatter(
            x=wl_ar, y=Rs,
            name='Rs',
            mode='lines',
            marker_color='rgba(0, 0, 255, .8)'
        ))
        fig.add_trace(go.Scatter(
            x=wl_ar, y=(Rp+Rs)/2,
            name='R(mean)',
            mode='lines',
            marker_color='rgba(0, 255, 0, .8)'
        ))

    # Set options common to all traces with fig.update_traces
    #fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    title_msg=f'Reflection at {round(inc_angle,1)}[deg]'
    fig.update_layout(title=title_msg,
                    yaxis_zeroline=True, xaxis_zeroline=True)
    #fig.update_layout(legend_title_text = "Contestant")
    fig.update_xaxes(title_text='Wavelength(nm)')
    fig.update_yaxes(title_text='Reflectance',range=[0, 1])


    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Colorimetry')

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
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
    illuminant = colour.SDS_ILLUMINANTS['D65']

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


    strRGB_p='#'+format(b_p[0], '02x')+format(b_p[1], '02x')+format(b_p[2], '02x')
    strRGB_s='#'+format(b_s[0], '02x')+format(b_s[1], '02x')+format(b_s[2], '02x')

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

    df=df.set_index('Wavelength(nm)')

    #st.write(df)
    #np.savetxt(".\\data\\temp\\data.csv",data,fmt='%.5f',delimiter=',') 

    csv = convert_df(df)

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    # YYYYMMDDhhmmss形式に書式化
    d = now.strftime('%Y%m%d%H%M%S')
    fname='data_'+d+'.csv'

    st.subheader('Download spectrum and color data')

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=fname,
        mime='text/csv',
    )


def disp_angle_scan():
    """
    inc_wl=600.0
    inc_angle_min=0.0
    inc_angle_max=80.0
    inc_angle_pitch=1.0

    """
    angle_ar=np.arange(inc_angle_min,inc_angle_max+inc_angle_pitch,inc_angle_pitch,dtype=float)
    Rp=np.zeros(len(angle_ar),dtype=float)
    Rs=np.zeros(len(angle_ar),dtype=float)

    Rp,Rs=calc_angle_reflectance(inc_wl,nk_fn_list,d_list,angle_ar)

    st.subheader('Reflectance at each incident angle')


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=angle_ar, y=Rp,
        name='Rp',
        mode='lines',
        marker_color='rgba(255, 0, 0, .8)'
    ))
    fig.add_trace(go.Scatter(
        x=angle_ar, y=Rs,
        name='Rs',
        mode='lines',
        marker_color='rgba(0, 0, 255, .8)'
    ))
    fig.add_trace(go.Scatter(
        x=angle_ar, y=Rp/Rs,
        name='Rp/Rs',
        mode='lines',
        marker_color='rgba(0, 255, 0, .8)'
    ))

    # Set options common to all traces with fig.update_traces
    #fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    title_msg=f'Reflection at {round(inc_wl,1)}[nm]'
    fig.update_layout(title=title_msg,
                    yaxis_zeroline=True, xaxis_zeroline=True)
    #fig.update_layout(legend_title_text = "Contestant")
    fig.update_xaxes(title_text='Incident angle [deg]')
    fig.update_yaxes(title_text='Reflectance',range=[0, 1])


    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Colorimetry')


    colour_wl_min=380
    colour_wl_max=780
    colour_wl_pitch=5

    colour_wl_ar=np.arange(colour_wl_min,colour_wl_max+colour_wl_pitch,colour_wl_pitch,dtype=float)
    inc_angle_ar=np.linspace(inc_angle_min,inc_angle_max,11,dtype=float)


    for idx,inc_angle in enumerate(inc_angle_ar):
        colour_Rp,colour_Rs=calc_reflectance(colour_wl_ar,nk_fn_list,d_list,inc_angle)
        colour_Rm=(colour_Rp+colour_Rs)/2.0
        sd_p = colour.SpectralDistribution(colour_Rp, name=f'Sample Rp{idx}')    
        sd_s = colour.SpectralDistribution(colour_Rs, name=f'Sample Rs{idx}')
        sd_m = colour.SpectralDistribution(colour_Rm, name=f'Sample Rm{idx}')
        sd_p.wavelengths=np.arange(colour_wl_min,colour_wl_max+colour_wl_pitch,colour_wl_pitch)
        sd_s.wavelengths=np.arange(colour_wl_min,colour_wl_max+colour_wl_pitch,colour_wl_pitch)
        sd_m.wavelengths=np.arange(colour_wl_min,colour_wl_max+colour_wl_pitch,colour_wl_pitch)

        # Convert to Tristimulus Values
        cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        illuminant = colour.SDS_ILLUMINANTS['D65']

        # Calculating the sample spectral distribution *CIE XYZ* tristimulus values.
        XYZ_p = colour.sd_to_XYZ(sd_p, cmfs, illuminant)
        XYZ_s = colour.sd_to_XYZ(sd_s, cmfs, illuminant)
        XYZ_m = colour.sd_to_XYZ(sd_m, cmfs, illuminant)
        RGB_p = colour.XYZ_to_sRGB(XYZ_p / 100)
        RGB_s = colour.XYZ_to_sRGB(XYZ_s / 100)
        RGB_m = colour.XYZ_to_sRGB(XYZ_m / 100)
        b_p=[]
        for v in RGB_p:
            b_p.append(np.clip(round(v*255),0,255))
        b_s=[]
        for v in RGB_s:
            b_s.append(np.clip(round(v*255),0,255))
        b_m=[]
        for v in RGB_m:
            b_m.append(np.clip(round(v*255),0,255))

        strRGB_p='#'+format(b_p[0], '02x')+format(b_p[1], '02x')+format(b_p[2], '02x')
        strRGB_s='#'+format(b_s[0], '02x')+format(b_s[1], '02x')+format(b_s[2], '02x')
        strRGB_m='#'+format(b_m[0], '02x')+format(b_m[1], '02x')+format(b_m[2], '02x')


        col1,col2,col3=st.columns(3)
        with col1:
            color_p = st.color_picker(f'Rp@{inc_angle:.2f}[deg]', strRGB_p,key=f'cp_Rp{idx}')
        with col2:
            color_s = st.color_picker(f'Rs@{inc_angle:.2f}[deg]', strRGB_s,key=f'cp_Rs{idx}')
        with col3:
            color_m = st.color_picker(f'R(mean)@{inc_angle:.2f}[deg]', strRGB_m,key=f'cp_Rm{idx}')



    nangle=len(angle_ar)
    data=np.concatenate([angle_ar.reshape([nangle,1]),Rp.reshape([nangle,1]),Rs.reshape([nangle,1])],1)
    
    df=pd.DataFrame(data)
    df.columns = ['Angle(deg)', 'Rp', 'Rs']

    df=df.set_index('Angle(deg)')

    # #st.write(df)
    # #np.savetxt(".\\data\\temp\\data.csv",data,fmt='%.5f',delimiter=',') 

    csv = convert_df(df)

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    # YYYYMMDDhhmmss形式に書式化
    d = now.strftime('%Y%m%d%H%M%S')
    fname='data_'+d+'.csv'

    st.subheader('Download reflectance data')

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=fname,
        mime='text/csv',
    )







st.title('Optical film simulator')

st.sidebar.header('Light parameters')

calc_mode=st.sidebar.radio("Calculation mode",calc_mode_menu)

if calc_mode=='Wavelength Scan':

    inc_angle=st.sidebar.number_input('Angle of Incidence [deg]',min_value=0.0,max_value=89.0,value=0.0,step=0.1,format='%3.1f')
    spMenu=('Visible[380-780nm]','UV[200-400nm]','NIR[700-1000nm]','All[200-1000nm]','Any')
    wl_option=st.sidebar.selectbox('Spetrum range',spMenu)
    if wl_option==spMenu[0]:
        wl_min=380.0
        wl_max=780.0
        wl_pitch=5.0
    elif wl_option==spMenu[1]:
        wl_min=200.0
        wl_max=400.0
        wl_pitch=0.5
    elif wl_option==spMenu[2]:
        wl_min=700.0
        wl_max=1000.0
        wl_pitch=2.0
    elif wl_option==spMenu[3]:
        wl_min=200.0
        wl_max=1000.0
        wl_pitch=1.0

    if wl_option==spMenu[4]:
        wl_range=st.sidebar.slider('Wavelength range [nm]',min_value=200.0,max_value=1000.0,value=(wl_min,wl_max),step=20.0,format='%.0f')
        if wl_range:
            wl_min=wl_range[0]
            wl_max=wl_range[1]
        wl_pitch=st.sidebar.number_input('Wavelength pitch [nm]',min_value=0.1,max_value=10.0,value=wl_pitch,step=0.1,format='%3.1f')

else:
    inc_wl=st.sidebar.number_input('Wavelength(nm)',min_value=0.0,max_value=1000.0,value=inc_wl,step=0.1,format='%3.1f')
    inc_angle_range=st.sidebar.slider('Incident angle range [deg]',min_value=0.0,max_value=85.0,value=(inc_angle_min,inc_angle_max),step=5.0,format='%.0f')
    if inc_angle_range:
        inc_angle_min=inc_angle_range[0]
        inc_angle_max=inc_angle_range[1]
    inc_angle_pitch=st.sidebar.number_input('Angle pitch [deg]',min_value=0.01,max_value=15.0,value=inc_angle_pitch,step=0.01,format='%3.2f')


st.sidebar.header('Atmosphere')
n_env=st.sidebar.number_input('Refractive index (air:1.00)',min_value=1.0,max_value=3.0,value=1.0,step=0.01,format='%3.2f')


st.header('Interference color of multilayer film')

st.subheader('Film stack')
nlayers=st.number_input('Number of layer',min_value=1,max_value=100,value=nlayers,step=1,format='%d')

nk_namelist=get_nk_list()
if len(nk_namelist)<1:
    st.error('nk list not find')


nk_idx_subst=nk_namelist.index('Silicon')
nk_idx_film=nk_namelist.index('SiO2')
# print('nk list',nk_namelist)

nk_name_list=[]
d_list=[]
nk_name_list.append(n_env)
d_list.append(np.Inf)

for num in range(nlayers):
    col1,col2=st.columns((2,1))
    label_layer=order_n(num+1)+' layer'
    with col1:
        nk_name=st.selectbox(label_layer,nk_namelist,index=nk_idx_film,key='L'+str(num+1))
        nk_name_list.append(nk_name)
    with col2:
        val=st.number_input('thickness[nm]',min_value=0.0,max_value=1e6,value=100.0,step=0.1,format='%g',key='T'+str(num+1))
        d_list.append(val)

nk_subst_name=st.selectbox('substrate',nk_namelist,index=nk_idx_subst,key='L0')
nk_name_list.append(nk_subst_name)
d_list.append(np.Inf)


# with st.expander("Direct input for Expert user"):
#     tmp_d_list=st.text_input("Thickness list (nm) ", d_list[1:-1])
#     tmp_d_list=eval(tmp_d_list)
#     if tmp_d_list!=d_list[1:-1]:
#         d_list[1:-1]=tmp_d_list
#         #st.write(d_list)
#     tmp_nk_name_list=st.text_input("Material list", nk_name_list[1:-1])
#     tmp_nk_name_list=eval(tmp_nk_name_list)
#     if tmp_nk_name_list!=nk_name_list[1:-1]:
#         nk_name_list=[]
#         nk_name_list.append(n_env)
#         nk_name_list.extend(tmp_nk_name_list)
#         nk_name_list.append(nk_subst_name)
#         st.write(nk_name_list)



nk_fn_list=make_nk_fn(nk_name_list)

if calc_mode=='Wavelength Scan':
    disp_wavelength_scan()
else:
    disp_angle_scan()


