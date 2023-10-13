import matplotlib.pyplot as plt         # библиотека для графиков
import pandas as pd                     # библиотека для обработки табличных данных
import math                             # библиотека для математики
import numpy as np
import tabulate
import pip
pip.main(["install", "openpyxl"])


cols = [1, 2, 3]

data = pd.read_excel('./lab4.xlsx', usecols = (cols))     # чтение файла
data.head()

#Про индексацию в двумерных массивах
#Строка 0 -- 440нм
#Строка 1 -- 505нм
#Строка 2 -- 525нм
#Строка 3 -- 580нм
#Строка 4 -- 595нм

j = data.iloc[:,0].tolist()
j = list(map(float, j))

arrofj = [0] * 11
for i in range(0,5):
    arrofj[i] = [0] * 11

i = 0

for k in range(0,4,1): #поменять на 5
    while not (i == 10 + 11 * k):
        arrofj[k][i - 11 * k] = j[i]
        i += 1

arrofj[0][10] = j[10]

#######################

alphaplus = data.iloc[:,1].tolist()
alphaplus = list(map(float, alphaplus))

arrofalphaplus = [0] * 11
for i in range(0,5):
    arrofalphaplus[i] = [0] * 11

i = 0

for k in range(0,4,1): #поменять на 5
    while not (i == 10 + 11 * k):
        arrofalphaplus[k][i - 11 * k] = alphaplus[i]
        i += 1

arrofalphaplus[0][10] = alphaplus[10]

#########################################

alphaminus = data.iloc[:,2].tolist()
alphaminus = list(map(float, alphaminus))

arrofalphaminus = [0] * 11
for i in range(0,5):
    arrofalphaminus[i] = [0] * 11

i = 0

for k in range(0,4,1): #поменять на 5
    while not (i == 10 + 11 * k):
        arrofalphaminus[k][i - 11 * k] = alphaminus[i]
        i += 1

arrofalphaminus[0][10] = alphaminus[10]

######################################

deltaalpha = [0] * 11
for i in range(0,5):
    deltaalpha[i] = [0] * 11

for i in range(0, 4):
    for k in range(0,10):
        deltaalpha[i][k] = 60 * (arrofalphaplus[i][k] - arrofalphaminus[i][k])

########################################

b = [0] * 11
for i in range(0,5):
    b[i] = [0] * 11

for i in range(0, 4):
    for k in range(0,10):
        b[i][k] = 77.8 * arrofj[i][k]

#########################################

psi = [0] * 11
for i in range(0,5):
    psi[i] = [0] * 11

for i in range(0, 4):
    for k in range(0,10):
        psi[i][k] = (deltaalpha[i][k] / 2) * math.pi / (60 * 180)


########################################

#df = pd.DataFrame({'Lambda = 440 nm': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                    'J, A' : arrofj[0],
#                    'Alpha+' : arrofalphaplus[0],
#                    'Alpha-': arrofalphaminus[0],
#                    'B, Tl': b[0],
#                    'Delta Alpha': deltaalpha[0],
#                    } )
#print(df.to_markdown())

x440 = np.array(b[0])
y440 = np.array(psi[0])

A440 = np.vstack([x440, np.ones(len(x440))]).T
m440, c440 = np.linalg.lstsq(A440, y440, rcond=None)[0]

##############################################

x505 = np.array(b[1])
y505 = np.array(psi[1])

A505 = np.vstack([x505, np.ones(len(x505))]).T
m505, c505 = np.linalg.lstsq(A505, y505, rcond=None)[0]

############################################

x525 = np.array(b[2])
y525 = np.array(psi[2])

A525 = np.vstack([x525, np.ones(len(x525))]).T
m525, c525 = np.linalg.lstsq(A525, y525, rcond=None)[0]

#################################################

x580 = np.array(b[3])
y580 = np.array(psi[3])

A580 = np.vstack([x580, np.ones(len(x580))]).T
m580, c580 = np.linalg.lstsq(A580, y580, rcond=None)[0]

v = [m440*1000, m505*1000 , m525*1000, m580*1000] #595



lambdas = [440*10**-9, 505*10**-9, 525*10**-9, 580*10**-9] #595

omega = []

for val in lambdas:
    omega.append(2 * math.pi * 3 * 10**8 / val)

omega2 = np.sqrt(omega)

print(v)

print(omega2)

x = np.array(omega2)
y = np.array(v)

A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]

print(m)

otvet = input("Что вывести? (1 - график для 440нм, 2 - график для 505нм, 3 - график для 525нм, 4 - график для 580нм, 5 - график зависимости постоянной Верде от частоты волны, 0 - выход)\n")
while True:
    if otvet == '1':

        figure, axis = plt.subplots(1, figsize = (15, 10))
        axis.scatter(b[0], psi[0])
        plt.xlim(0, 230)
        plt.ylim(-75, 2)
        plt.xlabel("B, Tl")
        plt.ylabel("Psi")
        plt.grid(color='lightgray', linestyle='--')
        plt.plot(x440, m440*x440 + c440, color = "green", alpha=0.4)
        plt.title('Рис. 1 Зависимость фарадеевского вращения от индукции магнитного поля(Lambda = 440нм)')

    elif otvet == '2':

        figure, axis = plt.subplots(1, figsize = (15, 10))
        axis.scatter(b[1], psi[1])
        plt.xlim(0, 230)
        plt.ylim(-400, 140)
        plt.xlabel("B, Tl")
        plt.ylabel("Psi")
        plt.grid(color='lightgray', linestyle='--')
        plt.plot(x505, m505 * x505 + c505, color="green", alpha=0.4)
        plt.title('Рис. 2 Зависимость фарадеевского вращения от индукции магнитного поля(Lambda = 505нм)')

    elif otvet == '3':

        figure, axis = plt.subplots(1, figsize = (15, 10))
        axis.scatter(b[2], psi[2])
        plt.xlim(0, 230)
        plt.ylim(-155, 10)
        plt.xlabel("B, Tl")
        plt.ylabel("Psi")
        plt.grid(color='lightgray', linestyle='--')
        plt.plot(x525, m525 * x525 + c525, color="green", alpha=0.4)
        plt.title('Рис. 3 Зависимость фарадеевского вращения от индукции магнитного поля(Lambda = 525нм)')

    elif otvet == '4':

        figure, axis = plt.subplots(1, figsize = (15, 10))
        axis.scatter(b[3], psi[3])
        plt.xlim(0, 230)
        plt.ylim(-325, 60)
        plt.xlabel("B, Tl")
        plt.ylabel("Psi")
        plt.grid(color='lightgray', linestyle='--')
        plt.plot(x580, m580 * x580 + c580, color="green", alpha=0.4)
        plt.title('Рис. 4 Зависимость фарадеевского вращения от индукции магнитного поля(Lambda = 580нм)')
    elif otvet == '5':

        figure, axis = plt.subplots(1, figsize = (15, 10))
        axis.scatter(omega2, v)
        plt.xlim(56500000, 66000000)
        plt.ylim(-0.7, 0.1)
        plt.xlabel("w^2")
        plt.ylabel("V")
        plt.grid(color='lightgray', linestyle='--')
        plt.plot(x, m * x + c, color="green", alpha=0.4)
        plt.title('Рис. 5 Зависимость постоянной Верде от квадрата частоты волны')



    elif otvet == '0':
        break
    else:
        print("Ты дебил?")
    plt.show()
    otvet = input("Что вывести? (1 - график для 440нм, 2 - график для 505нм, 3 - график для 525нм, 4 - график для 580нм, 5 - график зависимости постоянной Верде от частоты волны, 0 - выход)\n")