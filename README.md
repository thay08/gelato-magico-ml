# gelato-magico-ml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Simulando dados
data = {'Temperatura': [30, 25, 20, 35, 28, 22],
        'Vendas': [150, 100, 80, 200, 130, 90]}
df = pd.DataFrame(data)

# Dividindo os dados
X = df[['Temperatura']]
y = df['Vendas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazendo previsões
predictions = model.predict(X_test)

# Visualizando os resultados
plt.scatter(X, y, color='blue')
plt.plot(X_test, predictions, color='red')
plt.xlabel('Temperatura')
plt.ylabel('Vendas')
plt.title('Previsão de Vendas de Sorvete')
plt.show()
