# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:23:10 2017

@author: pfierens
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import fft

cap = cv2.VideoCapture('2017-09-14 21.53.59.mp4')

# if not cap.isOpened():
#    print("No lo pude abrir")
#    return

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

r = np.zeros((1, length))
g = np.zeros((1, length))
b = np.zeros((1, length))

k = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        r[0, k] = np.mean(frame[330:360, 610:640, 0])
        g[0, k] = np.mean(frame[330:360, 610:640, 1])
        b[0, k] = np.mean(frame[330:360, 610:640, 2])
    #        print(k)
    else:
        break
    k = k + 1

cap.release()
cv2.destroyAllWindows()

n = 1024
f = np.linspace(-n / 2, n / 2 - 1, n) * fps / n


r = r[0, 0:n] - np.mean(r[0, 0:n])
g = g[0, 0:n] - np.mean(g[0, 0:n])
b = b[0, 0:n] - np.mean(b[0, 0:n])

sta = time.perf_counter()

R = np.abs(np.fft.fftshift(np.fft.fft(r))) ** 2
G = np.abs(np.fft.fftshift(np.fft.fft(g))) ** 2
B = np.abs(np.fft.fftshift(np.fft.fft(b))) ** 2

end = time.perf_counter()


sta2 = time.perf_counter()

R2 = np.abs(np.fft.fftshift(fft.fft_opt(r, len(r), 1, 0))) ** 2
G2 = np.abs(np.fft.fftshift(fft.fft_opt(g, len(g), 1, 0))) ** 2
B2 = np.abs(np.fft.fftshift(fft.fft_opt(b, len(b), 1, 0))) ** 2

# R2 = np.abs(np.fft.fftshift(fft.dft(r))) ** 2
# G2 = np.abs(np.fft.fftshift(fft.dft(g))) ** 2
# B2 = np.abs(np.fft.fftshift(fft.dft(b))) ** 2

end2 = time.perf_counter()

sta3 = time.perf_counter()

R3 = np.abs(np.fft.fftshift(fft.fft_iter_opt(r))) ** 2
G3 = np.abs(np.fft.fftshift(fft.fft_iter_opt(g))) ** 2
B3 = np.abs(np.fft.fftshift(fft.fft_iter_opt(b))) ** 2

end3 = time.perf_counter()


print("Tiempo de corrida np.fft: {}".format(end - sta))
print("Tiempo de corrida dft: {}".format(end2 - sta2))
print("Tiempo de corrida fft_iter_opt: {}".format(end3 - sta3))

print(np.fft.fft(r) ** 2)
print(fft.fft_opt(r, len(r), 1, 0) ** 2)
print(fft.fft_iter_opt(r) ** 2)

plt.plot(60 * f, R, "r")
plt.xlim(0, 200)

plt.plot(60 * f, G, "g")
plt.xlim(0, 200)

plt.plot(60 * f, B, "b")
plt.xlim(0, 200)

plt.xlabel("frecuencia [1/minuto]")
# plt.figure()
# plt.plot(60 * f, R2, "r")
# plt.xlim(0, 200)
#
# plt.plot(60 * f, G2, "g")
# plt.xlim(0, 200)
#
# plt.plot(60 * f, B2, "b")
# plt.xlim(0, 200)
#
# plt.xlabel("frecuencia 2 [1/minuto]")

plt.figure()
plt.plot(60 * f, R3, "r")
plt.xlim(0, 200)

plt.plot(60 * f, G3, "g")
plt.xlim(0, 200)

plt.plot(60 * f, B3, "b")
plt.xlim(0, 200)

plt.xlabel("frecuencia 2 [1/minuto]")

plt.show()

print("Frecuencia cardíaca: ", abs(f[np.argmax(G)]) * 60, " pulsaciones por minuto")
