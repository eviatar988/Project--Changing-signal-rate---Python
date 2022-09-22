import math
import numpy as np
import cmath
import matplotlib.pyplot as plt
import soundfile as sf
from matplotlib import pyplot as plt
import sounddevice as sd

#Eviatar Cohen - 205913858
#Hai Moyal - 315669739

C = 8
N1 = 9

#COS FUNC
Ak_1 = np.zeros((C), dtype=np.complex_)

for n in range(C):
    Ak_1[n] = math.cos((2 * math.pi * n)/C)

#windowFunc
AK_2 = np.zeros((20 * N1), dtype=np.complex_)
AK_2[40] = 1
print('# window function')
for n in range(40):
    AK_2[n] = 1
    AK_2[159 - n] = 1

def FourierCoeffGen(signal):

    FourierCoeff = 0
    n_signal = len(signal)
    w_0 = 2*math.pi/n_signal
    a = np.zeros((n_signal), dtype=np.complex_)
    for k in range(n_signal):
        x = 0
        for num in range(n_signal):
            x = x + signal[num] * cmath.exp(0-1j * k * w_0 * num)
        x = x / n_signal
        a[k] = x
    FourierCoeff = a
    return FourierCoeff

def  DiscreteFourierSeries(FourierCoeff):
    signal = 0
    n_signal = len(FourierCoeff)
    b = np.zeros((n_signal), dtype=np.complex_)
    w_0 = 2 * math.pi / n_signal
    for n in range(n_signal):
        x = 0
        for k in range(n_signal):
            x = x + FourierCoeff[k] * cmath.exp(1j * k * w_0 * n)
        b[n] = x
    signal = b
    return signal


d = FourierCoeffGen(Ak_1)
h = DiscreteFourierSeries(d)
y_Point_1 = h
x_Point_1 = [i for i in range(len(y_Point_1))]
print(x_Point_1, y_Point_1)
plt.stem(x_Point_1, y_Point_1)
plt.show()

d = FourierCoeffGen(AK_2)
h = DiscreteFourierSeries(d)
y_Point_2 = h
x_Point_2 = [i for i in range(len(y_Point_2))]
plt.stem(x_Point_2, y_Point_2)
plt.show()
