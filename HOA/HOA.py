import sys
sys.path.append('/usr/local/lib/python3.9/site-packages')

import green_function as gr
import numpy as np
import scipy as sp
import function as fu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import my_function as mf
from scipy.io.wavfile import read, write
import pickle
import colormaps as cmaps

## 初期条件 ##############
cv = 344        #音速
fs = 48000      #サンプリング周波数
Nfft = 1024     #フーリエ点数
N_ext = 6       #展開次数, 仮想スピーカ及びマイク数: (N_ext+1)**2
N_int = 4       #展開次数
freq = 500      #対象周波数
enc = 1         #0:Open sphere encode, 1:Cardioid encode
face = 0        #0:Front, 1:Back
move = 1        #仮想音源構築位置に関するパラメータ (x,y,z)=(move, move, 0)

if enc == 0:
    dir = 'OS/'
elif enc == 1:
    dir = 'CS/'
fdir = str(freq)+'hz/'

if face == 0:
    angle = '45'
    radiation = 'front/'
    rng = 0.6
elif face == 1:
    angle = '225'
    radiation = 'back/'
    if freq == 250:
        rng = 0.4
    else:
        rng = 0.2

print(fdir+dir+radiation)

lambda_ = cv/freq
omega = 2 * np.pi * freq    #角周波数
k = omega/cv                #波数
rad = (N_int*cv)/(2*np.pi*freq)

Plot_range = 2.1    #プロットする領域
space = 0.05        #空間離散化幅
X = np.arange(-Plot_range,Plot_range+space,space)
Y = np.arange(-Plot_range,Plot_range+space,space)
################################################

#プロット調整用
LWidth = 6      #degault: 2, 再現領域とスピーカ半径の線の太さ
Fsize = 24      #default: 14, フォントサイズ
Lpad = 22       #default: 18, カラーバーラベルと図の距離
Fsize_ = 16     #default: 14, 横軸の単位の大きさ

## 放射指向特性の読み込み ################
import scipy.io
mat = scipy.io.loadmat('../anm/anm'+angle+'.mat')
data = mat['anm']
data = np.array(data)
num = int((freq+20)/(fs/Nfft))
Anm = data[0:(N_ext+1)**2,num] #5(250hz),11(500hz),21(1000hz)
################################################

## 仮想マイクの座標定義 ##############
r_mic = 0.05    #マイク半径
num_mic = (N_ext+1)**2    #マイク数
mic_xyz = fu.EquiDistSphere((N_ext+1)**2) #マイク配置角度
mic_xyz = mic_xyz * r_mic               #マイク配置座標

azi_mic, ele_mic, rr_mic = fu.cart2sph(mic_xyz[:,0],mic_xyz[:,1],mic_xyz[:,2])
ele_mic1 = np.pi/2 - ele_mic
################################################

## 重み算出 ##############
w = np.array([])  #各マイクにおける音圧
for i in range(num_mic):
    w_ = mf.calcDirWeights2([move,move,0],mic_xyz[i])
    w = np.append(w, w_)
################################################

## 仮想マイクの音圧算出 ##############
mic_sig = np.array([])  #各マイクにおける音圧
mic_xyz[:,0] = mic_xyz[:,0] - move
mic_xyz[:,1] = mic_xyz[:,1] - move

if enc == 0:
    for i in range(num_mic):
        mic_sig_ = Anm @ fu.calcPext(N_ext,k,mic_xyz[i,0],mic_xyz[i,1],mic_xyz[i,2])
        mic_sig = np.append(mic_sig, mic_sig_)
elif enc == 1:
    for i in range(num_mic):
        mic_sig_ = Anm @ fu.calcPext(N_ext,k,mic_xyz[i,0],mic_xyz[i,1],mic_xyz[i,2])
        mic_sig = np.append(mic_sig, mic_sig_)
    mic_sig = mic_sig*w #重み付け→カージオイドエンコード
################################################

## HOAエンコード(展開係数の算出) ##############
ambi_signal = fu.encode(N_ext,azi_mic,ele_mic1,k,r_mic,mic_sig,enc)
################################################

##　スピーカ座標の設定(22.2ch) ############################
r_sp = 1.9
sp_azi_deg = np.array([60, 300, 0, 120, 240, 30, 330, 180, 90, 270,
                60, 300, 0, 0, 120, 240, 90, 270, 180, 0, 60, 300])
sp_ele_deg = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 38, 38,
                90, 38, 38, 38, 38, 38, -28.3, -28.3, -28.3])
sp_azi = np.deg2rad(sp_azi_deg)
sp_ele = np.pi/2 - np.deg2rad(sp_ele_deg)
num_sp = 22
sp_r = np.full(num_sp, r_sp)
sp_sph = np.vstack([np.vstack([sp_azi,sp_ele]),sp_r]).T
sp_x = sp_r * np.sin(sp_ele) * np.cos(sp_azi)
sp_y = sp_r * np.sin(sp_ele) * np.sin(sp_azi)
sp_z = sp_r * np.cos(sp_ele)
sp_xyz = np.vstack([np.vstack([sp_x,sp_y]),sp_z]).T
################################################

## HOAデコード(展開係数の算出) ##############
ambi_signal_ = ambi_signal[0:(N_int+1)**2]

spkr_output = fu.decode(N_int,sp_azi,sp_ele,k,r_sp,lambda_,ambi_signal_)
################################################

## 表示上の位相調整項 ##############
if freq == 250:
    if face == 0:
        if enc == 0:
            delay = 6
        elif enc == 1:
            delay = 14
    elif face ==1:
        if enc == 0:
            delay = 86
        elif enc == 1:
            delay = 94
elif freq == 500:
    if face == 0:
        if enc == 0:
            delay = 92
        elif enc == 1:
            delay = 93
    elif face ==1:
        if enc == 0:
            delay = 60
        elif enc == 1:
            delay = 61
elif freq == 1000:
    if face == 0:
        if enc == 0:
            delay = 3
        elif enc == 1:
            delay = 3
    elif face ==1:
        if enc == 0:
            delay = 36
        elif enc == 1:
            delay = 36
else:
    print('Please adjust the delay parameter.')
    delay = 0
################################################

## ●●再現用22.2chスピーカによる音圧マップの算出 ##############
P_rep = np.zeros((X.size,Y.size), dtype = np.complex128)
for j in range(X.size):
    for i in range(Y.size):
        pos_r = np.array([X[j],Y[i],0])
        for l in range(num_sp):
            G_transfer = gr.green2(0.5,freq,cv,sp_xyz[l,:],pos_r) *np.exp(-1j * omega * delay/fs)
            P_rep[i,j] += G_transfer * spkr_output[l]
P_rep = 4*np.pi*P_rep
################################################

## 所望音場の算出 ############################
# move = 1
X_ = np.arange(-Plot_range-move,Plot_range+space-move,space)
Y_ = np.arange(-Plot_range-move,Plot_range+space-move,space)
P_des = np.zeros((X_.size,Y_.size), dtype = np.complex128)
for j in range(X_.size):
    for i in range(Y_.size):
        pos_r = np.array([X_[j],Y_[i],0])
        P_des[i,j] = np.conj(Anm @ fu.calcPext(N_ext,k,pos_r[0],pos_r[1],pos_r[2]))
################################################

## 誤差算出 ############################
temp = 0
NEave = 0
NE = np.zeros((X.size,Y.size), dtype = np.complex128)
for i in range(X.size):
    for j in range(Y.size):
        NE[i,j] = 10*np.log10(((np.abs(P_rep[i,j]-P_des[i,j]))**2)/((np.abs(P_des[i,j]))**2))
        # Xaxis = Plot_range*(i-((X.size/2)+1))/X.size
        # Yaxis = Plot_range*(j-((Y.size/2)+1))/Y.size
        if np.linalg.norm((X[i],Y[j]), ord=2)<rad:  #NREの平均算出用
            NEave += NE[i,j]
            temp += 1
            # print(temp)
NEave = round(np.real(NEave)/temp,1)
print('Average of NRE in reproduced region :',NEave,'[dB]')
################################################

## 所望音場のプロット ###########################
sweet_spot = patches.Circle(xy=(0, 0), radius=rad, ec='r',fill=False, lw=LWidth, ls=':')
r_s = 0.15
r_sp_ = patches.Rectangle(xy=(1-r_s, 1-r_s), width=2*r_s, height=2*r_s, angle=0,ec='k', fc='w', lw=2)
plt.figure()
plt.rcParams["font.size"] = Fsize
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = "stix"
ax = plt.axes()
im = ax.imshow(np.real(P_des), interpolation='gaussian',cmap=cmaps.parula,origin='lower',
               extent=[-Plot_range, Plot_range, -Plot_range, Plot_range],
               vmax=rng, vmin=-rng) #0.2,-0.2
ax.grid(False)
ax.add_patch(sweet_spot)
ax.add_patch(r_sp_)
# ax.scatter(sp_x[0:10],sp_y[0:10], c='white',s=100, marker='o', linewidths=3, edgecolors='black')  #マーカー(中層スピーカ)
plt.axis('scaled')
ax.set_xlabel(r"$\mathit{x}$ [m]")
ax.set_ylabel(r"$\mathit{y}$ [m]")
plt.xticks([-2, -1, 0, 1, 2], fontsize=Fsize_)
plt.yticks([-2, -1, 0, 1, 2], fontsize=Fsize_)

# plt.xticks([-0.4, -0.2, 0, 0.2, 0.4], fontsize=Fsize_)
# plt.yticks([-0.4, -0.2, 0, 0.2, 0.4], fontsize=Fsize_)
# ax.set_xlim(-0.45, 0.45)
# ax.set_ylim(-0.45, 0.45)

ax.set_aspect('equal')
cbar = plt.colorbar(im)
if face == 0:
    ticks = [-0.60, -0.45, -0.30, -0.15, 0, 0.15, 0.30, 0.45, 0.60] #front用
    cbar.set_ticks(ticks) #front用
cbar.set_label('Amplitude', labelpad=Lpad, rotation=270)
cbar.ax.tick_params(labelsize=Fsize_)
# plt.savefig('../sfr_plot/'+fdir+dir+radiation+'original_sound.pdf',bbox_inches="tight",dpi = 64,
#             facecolor = "white", tight_layout = True)
# plt.savefig('../sfr_plot/pptx/'+dir+radiation+'original_sound_.pdf',bbox_inches="tight",dpi = 64,
#             facecolor = "white", tight_layout = True)
# plt.savefig('../sfr_plot/pptx/original_sound_front.pdf',bbox_inches="tight",dpi = 64,
#             facecolor = "white", tight_layout = True)
plt.show()
################################################

## 再現用22.2chスピーカによる音圧マップのプロット ##############
r_sp = 1.9
sweet_spot = patches.Circle(xy=(0, 0), radius=rad, ec='r',fill=False, lw=LWidth, ls=':')
r_sp_ = patches.Circle(xy=(0, 0), radius=r_sp, ec='b', fill=False, lw=LWidth, ls=':')

plt.figure()
plt.rcParams["font.size"] = Fsize
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = "stix"
ax2 = plt.axes()
im2 = ax2.imshow(np.real(P_rep), interpolation='gaussian',cmap=cmaps.parula,origin='lower',
               extent=[-Plot_range, Plot_range, -Plot_range, Plot_range],
               vmax=rng, vmin=-rng)
ax2.grid(False)

ax2.add_patch(sweet_spot)
ax2.add_patch(r_sp_)
ax2.scatter(sp_x[0:10],sp_y[0:10], c='white',s=200, marker='o', lw=2, edgecolors='black', zorder=2)  #マーカー(中層スピーカ)

plt.axis('scaled')
ax2.set_xlabel(r"$\mathit{x}$ [m]")
ax2.set_ylabel(r"$\mathit{y}$ [m]")
plt.xticks([-2, -1, 0, 1, 2], fontsize=Fsize_)
plt.yticks([-2, -1, 0, 1, 2], fontsize=Fsize_)
ax2.set_aspect('equal')
cbar = plt.colorbar(im2)
if face == 0:
    ticks = [-0.60, -0.45, -0.30, -0.15, 0, 0.15, 0.30, 0.45, 0.60] #front用
    cbar.set_ticks(ticks) #front用
cbar.set_label('Amplitude', labelpad=Lpad, rotation=270)
cbar.ax.tick_params(labelsize=Fsize_)
# plt.savefig('../sfr_plot/'+fdir+dir+radiation+'reproduced_sound.pdf',bbox_inches="tight",dpi = 64,
#             facecolor = "white", tight_layout = True)
# plt.savefig('../sfr_plot/pptx/'+dir+radiation+'reproduced_sound.pdf',bbox_inches="tight",dpi = 64,
#             facecolor = "white", tight_layout = True)
plt.show()
################################################

## 誤差のプロット ##############
sweet_spot = patches.Circle(xy=(0, 0), radius=rad, ec='r',fill=False, lw=LWidth, ls=':')
r_sp_ = patches.Circle(xy=(0, 0), radius=r_sp, ec='b', fill=False, lw=LWidth, ls=':')

plt.figure()
plt.rcParams["font.size"] = Fsize
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = "stix"
ax3 = plt.axes()
im3 = ax3.imshow(np.real(NE), interpolation='gaussian',cmap=cm.pink_r,origin='lower',
               extent=[-Plot_range, Plot_range, -Plot_range, Plot_range],
               vmax=0, vmin=-50)
ax3.grid(False)

ax3.add_patch(sweet_spot)
ax3.add_patch(r_sp_)
ax3.scatter(sp_x[0:10],sp_y[0:10], c='white',s=200, marker='o', lw=2, edgecolors='black', zorder=2)  #マーカー(中層スピーカ)

plt.axis('scaled')
ax3.set_xlabel(r"$\mathit{x}$ [m]")
ax3.set_ylabel(r"$\mathit{y}$ [m]")
plt.xticks([-2, -1, 0, 1, 2], fontsize=Fsize_)
plt.yticks([-2, -1, 0, 1, 2], fontsize=Fsize_)
ax3.set_aspect('equal')
cbar = plt.colorbar(im3)
# cbar.set_label('Normalized error [dB]', labelpad=Lpad, rotation=270)
cbar.set_label('NRE [dB]', labelpad=Lpad, rotation=270)
cbar.ax.tick_params(labelsize=Fsize_)
# plt.savefig('../sfr_plot/'+fdir+dir+radiation+'reproduced_error.pdf',bbox_inches="tight",dpi = 64,
#             facecolor = "white", tight_layout = True)
# plt.savefig('../sfr_plot/pptx/'+dir+radiation+'reproduced_error.pdf',bbox_inches="tight",dpi = 64,
#             facecolor = "white", tight_layout = True)
plt.show()
