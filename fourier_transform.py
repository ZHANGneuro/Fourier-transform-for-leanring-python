
########################################################################
# Discrete Fourier Transform (DFT)
# method 1
########################################################################
import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from scipy.integrate import quad
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import random
import math
import cmath

Num_samples = 360
ampli_1 = 1
ampli_2 = 2
freq_xi_1 = 10
freq_xi_2 = 30
phas = 90  # in degree

raw_1 = []
for temp_t in range(Num_samples):
    fn = ampli_1 * math.sin(freq_xi_1 * 2*cmath.pi * float(temp_t) / Num_samples + 2*cmath.pi * phas / Num_samples)
    raw_1.append(fn)

raw_2 = []
for temp_t in range(Num_samples):
    fn = ampli_2 * math.sin(freq_xi_2 * 2*cmath.pi * float(temp_t) / Num_samples + 2*cmath.pi * phas / Num_samples)
    raw_2.append(fn)

raw_data = []
for temp_element in range(len(raw_1)):
    raw_data.append(raw_1[temp_element]+raw_2[temp_element])

# plot raw data
# x = np.linspace(0, len(raw_data), len(raw_data))
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# ax.plot(x, raw_data)
# plt.show()

def draw_plot():
    fig = plt.figure(figsize=(5, 5))
    x = np.linspace(0, len(imag_list), len(imag_list))
    title_subplot1 = "Real wave, freq = " + str(temp_xi) + "hz"
    title_subplot2 = "Imag wave, freq = " + str(temp_xi) + "hz"
    title_subplot3 = "Complex number, freq = " + str(temp_xi) + "hz"
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_ylim(-10, 10)
    plt.plot(x, real_list)
    plt.title(title_subplot1)
    ax2 = plt.subplot(3, 1, 2)
    ax2.set_ylim(-10, 10)
    plt.plot(x, imag_list)
    plt.title(title_subplot2)
    ax3 = plt.subplot(3, 1, 3)
    ax3.set_ylim(-10, 10)
    ax3.set_xlim(-10, 10)
    plt.plot(real_list, imag_list)
    plt.title(title_subplot3)
    plt.show()
    return fig, ax1, ax2, ax3,title_subplot1, title_subplot2, title_subplot3

# dft version of for loop
iter_freq = 40
list_loop = []
for temp_xi in range(iter_freq):
    Fm = 0.0
    real_list = []
    imag_list = []
    for temp_t in range(Num_samples):
        output_cur_freq = raw_data[temp_t] * cmath.exp( - 1j * temp_xi * 2*cmath.pi * temp_t / Num_samples )
        real_part = scipy.real(output_cur_freq)
        imag_part = scipy.imag(output_cur_freq)
        Fm += output_cur_freq
        real_list.append(real_part)
        imag_list.append(imag_part)
    list_loop.append(Fm / Num_samples)
    draw_plot()

# plot function
x = np.linspace(0, len(list_loop), len(list_loop))
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(x, list_loop)
plt.show()








########################################################################
# Discrete Fourier Transform (DFT)
# method 1 with animation
########################################################################
import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from scipy.integrate import quad
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import random
import math
import cmath

Num_samples = 360
ampli_1 = 1
ampli_2 = 2
freq_xi_1 = 10
freq_xi_2 = 30
phas = 90  # in degree

raw_1 = []
for temp_t in range(Num_samples):
    fn = ampli_1 * math.sin(freq_xi_1 * 2*cmath.pi * float(temp_t) / Num_samples + 2*cmath.pi * phas / Num_samples)
    raw_1.append(fn)

raw_2 = []
for temp_t in range(Num_samples):
    fn = ampli_2 * math.sin(freq_xi_2 * 2*cmath.pi * float(temp_t) / Num_samples + 2*cmath.pi * phas / Num_samples)
    raw_2.append(fn)

raw_data = []
for temp_element in range(len(raw_1)):
    raw_data.append(raw_1[temp_element]+raw_2[temp_element])

def draw_plot():
    fig = plt.figure(figsize=(5, 12))
    x = np.linspace(0, Num_samples, Num_samples)
    y = [0] * Num_samples
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_title(" ")
    ax1.set_ylim(-10, 10)
    sub_fig_1, = plt.plot(x, y)

    ax2 = plt.subplot(3, 1, 2)
    ax2.set_title(" ")
    ax2.set_ylim(-10, 10)
    sub_fig_2, = plt.plot(x, y)

    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title(" ")
    ax3.set_ylim(-10, 10)
    ax3.set_xlim(-10, 10)
    sub_fig_3, = plt.plot(x, y)

    plt.subplots_adjust(hspace= 0.5)
    return fig, ax1, ax2, ax3, sub_fig_1, sub_fig_2, sub_fig_3

def plot_ani(i):
    Fm = 0.0
    real_list = []
    imag_list = []
    title_subplot1 = "Real wave, freq = " + str(i) + "hz"
    title_subplot2 = "Imag wave, freq = " + str(i) + "hz"
    title_subplot3 = "Complex number, freq = " + str(i) + "hz"
    for temp_t in range(Num_samples):
        output_cur_freq = raw_data[temp_t] * cmath.exp( - 1j * i * 2*cmath.pi * temp_t / Num_samples )
        real_part = scipy.real(output_cur_freq)
        imag_part = scipy.imag(output_cur_freq)
        Fm += output_cur_freq
        real_list.append(real_part)
        imag_list.append(imag_part)
    x = np.linspace(0, Num_samples, Num_samples)
    sub_fig_1.set_ydata(real_list)
    sub_fig_2.set_ydata(imag_list)
    sub_fig_3.set_xdata(real_list)
    sub_fig_3.set_ydata(imag_list)
    ax1.set_title(title_subplot1)
    ax2.set_title(title_subplot2)
    ax3.set_title(title_subplot3)


iter_freq = 40
fig, ax1, ax2, ax3, sub_fig_1, sub_fig_2, sub_fig_3 = draw_plot()
ani = animation.FuncAnimation(fig, plot_ani, frames=iter_freq, interval=700, repeat=False)
ani.save('/Users/boo/Desktop/fourier_transform_example.gif', writer='imagemagick', dpi=80)















# ########################################################################
# # Fourier Transform using intergral
# # method 2
# ########################################################################
# import numpy
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy
# import scipy.fftpack
# from scipy.integrate import quad
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import matplotlib.patches as patches
# from matplotlib.patches import Rectangle
# from matplotlib.animation import FuncAnimation
# import random
# import math
# import cmath
#
# Num_samples = 360
# ampli = 10
# freq = 50
# phas = 10  # in degree
#
# def func_inter(x):
#     radi= float(x) / Num_samples * cmath.pi * 2
#     y_of_curve = ampli * math.sin(freq * radi + phas/360 * cmath.pi * 2)
#     area_for_current_freq = y_of_curve * cmath.exp( -1j * 2*cmath.pi * temp_xi * x/Num_samples )
#     return area_for_current_freq/Num_samples
#
# def real_func(x):
#     return scipy.real(func_inter(x))
#
# def imag_func(x):
#     return scipy.imag(func_inter(x))
#
# list_integral = []
# for temp_xi in range(Num_samples):
#     real_integral, err = quad(real_func, 0, Num_samples)
#     imag_integral, err = quad(imag_func, 0, Num_samples)
#     list_integral.append( real_integral + imag_integral*1j  )
#
# # plot function
# x = np.linspace(0, len(list_integral), len(list_integral))
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# ax.plot(x, list_integral)
# plt.show()






