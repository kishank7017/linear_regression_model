import pandas as pd
import matplotlib.pyplot as plt

data_1=pd.read_csv('LinearRegressionModel/score.csv')

# print(data)

# plt.scatter(data_1.Hours,data_1.Scores)
# plt.show()


def gradient_descent(m_now,c_now,points,L):
    m_gradient=0
    b_gradient=0

    n=len(points)

    for i in range(n):
        x=points.iloc[i].Hours
        y=points.iloc[i].Scores
        #derivative of loss of m
        m_gradient+= -(2 / n) * x * (y - (m_now * x + c_now))
        #derivative of loss of b
        b_gradient+= -(2 / n) * (y - (m_now * x + c_now))
    
    m= m_now - L * m_gradient
    c= c_now - L* b_gradient
    return m,c


m=0
c=0
L=0.0001
epochs=1000

for i in range (epochs):
    m,c=gradient_descent(m,c,data_1,L)

print(m,c)

plt.scatter(data_1.Hours,data_1.Scores)
plt.plot(list(range(0,10)),[m*x+c for x in range(0,10)],color='red')
plt.show()

