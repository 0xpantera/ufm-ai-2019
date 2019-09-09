#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch


# # Neural Networks

# * El uso de funciones de activacion no lineares como la diferencia clave entre modelos lineales
# * Los diferentes tipos de funciones de activacion
# * El modulo `nn` de PyTorch que contiene los bloques para construir NNs
# * Resolver un problema simple de un _fit_ lineal con una NN

# ## Neuronas artificiales
# 
# * Neural networks: entidades matematicas capaces de representar funciones complicadas a traves de una composicion de funciones mas simples.
# * Originalmente inspiradas por la forma en la que funciona nuestro cerebro.
# * El bloque de construccion basico es una neurona:
#     * Esencialmente una transformacion linear del input (e.g. multiplicacion del input por un numero, el _weight_, y la suma de una constante, el _bias_.
#     * Seguido por la aplicacion de una funcion no lineal (referida como la funcion de activacion)
#     * $o = f(w x + b)$
#     * x es nuestro input, w el _weight_ y b el _bias_. $f$ es la funcion de activacion.
#     * x puede ser un escalar o un vector de valores, w puede ser un escalar o una matriz, mientras que b es un escalar o un vector.
# * La expresion $o = f(w x + b)$ es una capa de neuronas, ya que representa varias neuronas a traves de los _weights_ y _bias_ multidimensionales

# $x_1 = f(w_0 x_0 + b_0)$
# 
# $x_2 = f(w_1 x_1 + b_1)$
# 
# $...$
# 
# $y = f(w_n x_n + b_n)$

# ### **dibujos**

# ## Funciones de activacion

# * Nuestro modelo anterior ya tenia una operacion lineal. Eso era el modelo entero.
# * El rol de la funcion de activacion es concentrar los _outputs_ de la operacion lineal precedente a un rango dado.
# * Si queremos asignar un _score_ al output del modelo necesitamos limitar el rango de numeros posibles para ese _score_
#     * `float32`
#     * $\sum wx + b$

# ### Que opciones tenemos?

# * Una opcion seria ponerle un limite a los valores del _output_.
#     * Cualquier cosa debajo de cero seria cero
#     * cualquier cosa arriba de 10 seria 10
#     * `torch.nn.Hardtanh`

# In[3]:


import math

math.tanh(-2.2) # camion


# In[4]:


math.tanh(0.1) # oso


# In[5]:


math.tanh(2.5) # perro


# ![Funciones de activacion](../assets/activaciones.png)

# * Hay muchas funciones de activacion.
# * Por definicion, las funciones de activacion:
#     * Son no lineales. Aplicaciones repetidas de $wx+b$ sin una funcion de activacion resultan en una polinomial. La no linealidad permite a la red aproximar funciones mas complejas.
#     * Son diferenciables, para poder calcular las gradientes a traves de ellas. Discontinuidades de punto como en `Hatdtanh` o `ReLU` son validas.
# * Sin esto, las redes caen a ser polinomiales complicadas o dificiles de entrenar.
# * Adicionalmente, las funciones:
#     * Tienen al menos un rango sensible, donde cambios no triviales en el input resultan en cambio no trivial correspondiente en el output
#     * Tienen al menos un rango no sensible (o saturado), donde cambios al input resultan en poco o ningun cambio en el output.
# * Por utlimo, las fuciones de activacion tienen al menos una de estas:
#     * Un limite inferior que se aproxima (o se encuentra) mientras el input tiende a negativo infinito.
#     * Un limite superior similar pero inverso para positivo infinito.
# * Dado lo que sabemos de como funciona back-propagation
#     * Sabemos que los errores se van a propagar hacia atras a traves de la activacion de manera mas efectiva cuando los inputs se encuentran dentro del rango de respuesta.
#     * Por otro lado, los errores no van a afectar a las neuornas para cuales el _input_ esta saturado debido a que la gradiente estara cercana a cero.

# ### En conclusion
# 
# * En una red hecha de unidades lineales + activaciones, cuando recibe diferentes _inputs_:
#     * diferentes unidades van a responder en diferentes rangos para los mismos inputs
#     * los errores asociados a esos inputs van a afectar a las neuronas operancio en el rango sensible, dejando a las otras unidades mas o menos igual en el proceso de aprendizaje. 
# * Juntar muchas operaciones lineales + unidades de activacion en paralelo y apilandolas una sobre otra nos provee un objeto matematico capaz de aproximar funciones complicadas. 
# * Diferentes combinaciones de unidades van a responder a inputs en diferentes rangos
#     * Esos parametros son relativamente faciles de optimizar a traves de SGD

# ### Dibujo graficas computacionales separadas

# In[7]:


import torch.nn as nn

linear_model = nn.Linear(1, 1)
linear_model(val_t_un)


# Todas las subclases de `nn.Module` tienen un metodo `call` definido. Esto permite crear una instancia de `nn.Linear` y llamarla como si fuera una funcion.
# 
# Llamar una instancia de `nn.Module` con un conjunto de argumetnos termina llamando un metodo llamado `forward` con esos mismos argumentos

# ### Implementacion de `Module.call`
# 
# (simplificado para claridad)

# In[8]:


def __call__(self, *input, **kwargs):
    for hook in self._forward_pre_hooks.values():
        hook(self, input)
        
    result = self.forward(*input, **kwargs)
    
    for hook in self._forward_hooks.values():
        hook_result = hook(self, input, result)
        # ...
        
    for hook in self._backward_hooks.values():
        # ...
        
        return result


# ### De regreso al modelo lineal

# In[9]:


import torch.nn as nn

linear_model = nn.Linear(1, 1)
linear_model(val_t_un)


# `nn.Linear` acepta tres argumentos:
# * el numero de input features: size del input = 1
# * numero de output features: size del outpu = 1
# * si incluye un bias o no (por default es `True`)

# In[10]:


linear_model.weight


# In[11]:


linear_model.bias


# In[12]:


x = torch.ones(1)
linear_model(x)


# * Nuestro modelo toma un input y produce un output
# * `nn.Module` y sus subclases estan diseniados para hacer eso sobre multiples muestras al mismo tiempo
# * Para acomodar multiples muestras los modulos esperan que la dimension 0 del input sea el numero de muestras en un _batch_
# * Cualquier module en `nn` esta hecho para producir outputs para un _batch_ de multiples inputs al mismo tiempo.
# * B x Nin
#     * B es el tamanio del _batch_
#     * Nin el numero de input features

# In[13]:


x = torch.ones(10, 1)
linear_model(x)


# Para un dataset de imagenes:
# * BxCxHxW

# In[14]:


t_c.size()


# In[6]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # Temperatura en grados celsios
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # Unidades desconocidas
t_c = torch.tensor(t_c).unsqueeze(1) # Agregamos una dimension para tener B x N_inputs
t_u = torch.tensor(t_u).unsqueeze(1) # Agregamos una dimension para tener B x N_inputs

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u


# In[15]:


import torch.nn as nn
import torch.optim as optim


params_old = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate_old = 1e-1
optimizer_old = optim.Adam([params_old], lr=learning_rate_old)


linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(
    linear_model.parameters(), # reemplazamos [params] con este metodo 
    lr=1e-2)


# ### linear_model.parameters()

# In[16]:


list(linear_model.parameters())


# In[17]:


def training_loop(model, n_epochs, optimizer, loss_fn, train_x, val_x, train_y, val_y):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_x) # ya no tenemos que pasar los params
        train_loss = loss_fn(train_t_p, train_y)
        
        with torch.no_grad(): # todos los args requires_grad=False
            val_t_p = model(val_x)
            val_loss = loss_fn(val_t_p, val_y)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss}")


# In[18]:


linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

training_loop(
    n_epochs=3000,
    optimizer=optimizer,
    model=linear_model,
    loss_fn=nn.MSELoss(), # Ya no estamos usando nuestra loss function hecha a mano
    train_x = train_t_un,
    val_x = val_t_un,
    train_y = train_t_c,
    val_y = val_t_c)

print()
print(linear_model.weight)
print(linear_model.bias)


# ## Finalmente un Neural Network

# * Ultimo paso: reemplazar nuestro modelo lineal
# * No va a ser mejor
# * Lo unico que vamos a cambiar va a ser el modelo
# * Un simple NN:
#     * Una capa lineal
#     * Activacion
#     * "hidden layers"

# In[19]:


seq_model = nn.Sequential(
                nn.Linear(1, 13), # El 13 es arbitrario
                nn.Tanh(),
                nn.Linear(13, 1) # Este 13 debe hacer match con el primero
            )

seq_model


# * El resultado final es un modelo que toma los inputs esperados por el primer modulo (_layer_)
# * Pasa los outputs intermedios al resto de los modulos
# * Produce un output retornado por el ultimo modulo

# In[20]:


[param.size() for param in seq_model.parameters()]


# * Estos son los parametros que el optimizador va a recibir
# * Al llamar `backward()` todos los parametros se van a llenar con su `grad`
# * El optimizador va a actualizar el valor de `grad` durante `optimizer.step()`

# In[21]:


for name, param in seq_model.named_parameters():
    print(name, param.size())


# In[22]:


from collections import OrderedDict

named_seq_model = nn.Sequential(OrderedDict([
        ('hidden_linear', nn.Linear(1, 8)),
        ('hidden_activation', nn.Tanh()),
        ('output_linear', nn.Linear(8, 1))
]))

seq_model


# In[23]:


for name, param in named_seq_model.named_parameters():
    print(name, param.size())


# In[24]:


named_seq_model.output_linear.bias


# Util para inspeccionar parametros o sus gradientes.

# In[ ]:


optimizer = optim.SGD(seq_model.parameters(), lr=1e-3)

training_loop(
    n_epochs=5000,
    optimizer=optimizer,
    model=seq_model,
    loss_fn=nn.MSELoss(), # Ya no estamos usando nuestra loss function hecha a mano
    train_x = train_t_un,
    val_x = val_t_un,
    train_y = train_t_c,
    val_y = val_t_c)

print('output', seq_model(val_t_un))
print('answer', val_t_c)
print('hidden', seq_model.hidden_linear.weight.grad)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
z = x + y
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end))


# Tambien podemos evaluar el modelo en toda la data y ver que tan diferente es de una linea:

# In[26]:


from matplotlib import pyplot as plt

t_range = torch.arange(20., 90.).unsqueeze(1)

fig = plt.figure(dpi=600)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
plt.show()


# ## Subclassing nn.Module
# 
# * sublcassing `nn.Module` nos da mucha mas flexibilidad.
# * La interface especifica que como minimo debemos definir un metodo `forward` para la subclase
#     * `forward` toma el input al model y regresa el output
# * Si usamos las operaciones de `torch`, `autograd` se encarga de hacer el `backward` pass de forma automatica
# 
# * Normalmente vamos a definir los submodulos que usamos en el metodo `forward` en el constructor
#     * Esto permite que sean llamados en `forward` y que puedan mantener sus parametros a durante la existencia de nuestro modulo

# In[27]:


class SubclassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13, 1)

        
    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        #activated_t = self.hidden_activation(hidden_t) if random.random() > 0.5 else hidden_t
        output_t = self.output_linear(activated_t)

        return output_t

    
subclass_model = SubclassModel()
subclass_model


# * Nos permite manipular los outputs de forma directa  y transformarlo en un tensor BxN
# * Dejamos la dimension de batch como -1 ya que no sabemos cuantos inputs van a venir por batch

# * Asignar una instancia de `nn.Module` a un atributo en un `nn.Module` registra el modulo como un submodulo.
# * Permite a `Net` acceso a los `parameters` de sus submodulos sin necesidad de hacerlo manualmente

# In[28]:


numel_list = [p.numel() for p in subclass_model.parameters()]
sum(numel_list), numel_list


# **Lo que paso**
# 
# * `parameters()` investiga todos los submodulos asignados como atributos del constructor y llama `parameters` de forma recursiva.
# * Al accesar su atributo `grad`, el cual va a ser llenado por el `autograd`, el optimizador va a saber como cambiar los parametros para minimizar el _loss_

# In[29]:


for type_str, model in [('seq', seq_model), ('named_seq', named_seq_model), ('subclass', subclass_model)]:
    print(type_str)
    for name_str, param in model.named_parameters():
        print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))

    print()


# In[30]:


class SubclassFunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 14)
        self.output_linear = nn.Linear(14, 1)

        
    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = torch.tanh(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t


func_model = SubclassFunctionalModel()
func_model


# ## Ejercicios

# * Experimenten con el numero de neuronas en el modelo al igual que el learning rate.
#     * Que cambios resultan en un output mas lineal del modelo?
#     * Pueden hacer que el modelo haga un overfit obvio de la data?
#     
# * Cargen la [data de vinos blancos](https://archive.ics.uci.edu/ml/datasets/wine+quality) y creen un modelo con el numero apropiado de inputs
#     * Cuanto tarda en entrenar comparado al dataset que hemos estado usando?
#     * Pueden explicar que factores contribuyen a los tiempos de entrenamiento?
#     * Pueden hacer que el _loss_ disminuya?
#     * Intenten graficar la data

# In[84]:


import time

start = time.time()
seq_model = nn.Sequential(
                nn.Linear(1, 3000), 
                nn.Tanh(),
                nn.Linear(3000, 1) # 
            )

optimizer = optim.SGD(seq_model.parameters(), lr=1e-4)

training_loop(
    n_epochs=9000,
    optimizer=optimizer,
    model=seq_model,
    loss_fn=nn.MSELoss(), # Se utiliza la función pytorch, no la generada manualmente
    train_x = train_t_un,
    val_x = val_t_un,
    train_y = train_t_c,
    val_y = val_t_c)

end = time.time()
print(end - start)


# # Experimentando con el numero de neuronas en el modelo al igual que el learning rate.
# ### Que cambios resultan en un output mas lineal del modelo?
# 
# A un mayor número de repeticiones y neuronas la función de perdida reduce su tamaño, para lo cual se le atribuye mayor exactitud a un aumento del número de neuronas. Un numero de learning rate muy pequeño puede no implicar mejoras significativas, sin embargo el mejor resultado en la función loss se observa en el nivel 1e-4 debido a que cualquier denotación mayor no implicaba un aporte al modelo. 
# 
# 
# 
# ### Pueden hacer que el modelo haga un overfit obvio de la data?
# 
# Un overfit obvio se puede generar al crear demasiadas neuronas y al fijar el learning rate en el monto más alto permitido
# 

# In[62]:




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[63]:


whine_df = pd.read_csv("winequality-white.csv", sep=";")
whine_df.head()


# In[65]:





corr = whine_df.corr()
 
corr


# In[74]:


X = whine_df['alcohol']
y = whine_df['quality']


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[76]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train = torch.tensor(X_train).unsqueeze(1)
X_test = torch.tensor(X_test).unsqueeze(1)
y_train = torch.tensor(y_train).unsqueeze(1)
y_test = torch.tensor(y_test).unsqueeze(1)


# In[83]:


import time

start = time.time()
seq_model = nn.Sequential(
                nn.Linear(1, 1000), 
                nn.Tanh(),
                nn.Linear(1000, 1) 
            )

optimizer = optim.SGD(seq_model.parameters(), lr=1e-4)

training_loop(
    n_epochs=5000,
    optimizer=optimizer,
    model=seq_model,
    loss_fn=nn.MSELoss(), 
    train_x = X_train.float(),
    val_x = X_test.float(),
    train_y = y_train.float(),
    val_y = y_test.float())



end = time.time()
print(end - start)


# ## Cuanto tarda en entrenar comparado al dataset que hemos estado usando?
# 
# #### Este dataset considerablemente toma más tiempo en entrenarse, se utilizo la función time.time para evaluar el tiempo elapsado y principalmente puedo atribuirlo al número de datos implicados en este set y la exactitud requerida. El primero modelo se entreno en 11 segundos mientras que el segundo demostro 422 segundos 
# 
# 
# ## Pueden explicar que factores contribuyen a los tiempos de entrenamiento?
# 
# #### Los factores que explican el tiempo de entrenamiento son el numero de neuronas implicado en la red neuronal predominantemente, el tamaño del learning rate planteado y el número de repeticiones o epochs. Todas estas variables implican un aumento en la exactitud. 
# 
# ## Pueden hacer que el loss disminuya?
# 
# #### Al involucrar en el modelo las variables más correlacionadas como el alcohol para predecir la calidad del vino se reduce la función de perdida, al igual que al aumentar el learning rate al máximo disponible. 
# 
# ## Intenten graficar la data
# 

# In[ ]:


get_ipython().system('jupyter nbconvert --to script Juarez_Boris_Tarea4_neural_networks.ipynb')

