#!/usr/bin/env python
# coding: utf-8

# # Aprendizaje

# In[46]:


import numpy as np
import torch
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# * Crear un algoritmo que _aprende_ de la data
# * Hacer un _fit_ a la data
# * El proceso: Una funcion con un numero de parametros desconocidos cuyos valores son estimados de la data
# * Un modelo

# Pytorch esta diseniado para faciliatar la creacion de modelos para los cuales las derivadas del error con respecto a los parametros pueden ser expresadas de forma analitica.

# ## El aprendizaje es la estimacion de parametros
# 
# * Data
# * Escoger un modelo
# * Estimar los parametros del modelo para tener buenas predicciones sobre data nueva.

# ## El problema
# 
# Tenemos un termometro que no ensenia las unidades en las cuales muestra la temperatura.
# 
# * data: lecturas del termometro y los valores correspondientes en una unidad conocida.
# * Escoger un modelo
# * Ajustar los parametros iterativamente hasta ue la medida del error sea lo suficientemente baja

# In[47]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # Temperatura en grados celsios
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # Unidades desconocidas
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# ### Escogiendo nuestro primer modelo
# 
# t_c = w * t_u + b
# 
# **Spoiler**: Sabemos que un modelo linear es el modelo correcto.
# 
# * Tenemos un modelo con parametros desconocidos: $w$ y $b$
# * Tenemos que estimar esos parametros para que el error entre $\hat{y}$ y $y$ sea lo mas pequenio posible
# * Todavia no hemos definido una medida de ese error.
#     * Esta medida, _loss function_, deberia ser alta si el error es alto e idealmente deberia ser lo mas bajo posible cuando haya un match perfecto.
# * Nuestro proceso de optimizacion deberia encontrar $w$ y $b$ para que el _loss function_ este en un minimo.

# ### Loss function
# 
# * Una funcion que calcula un valor numerico que el proceso de optimizacion va a intentar minimizar.
# * El calculo del _loss_ normalmente involucra tomar la diferencia entre el output deseado para alguna muestra de entrenamiento y el verdadero output producido por el modelo cuando ve esos outputs.
# * En nuestro caso: `t_p - t_c`
# * Nuestra _loss_function_ deberia siempre ser positiva
# * Conceptualmente un _loss function_ es una forma de priorizar cuales errores de nuestro training sample arreglar, para que los ajustes a los parametros resulten en ajustes a los outputs para las muestras con mayor peso.
# 
# 
# Ejemplos:
# * $|t_p - t_c|$
# * $(t_p - t_c)^2$

# ## Problema a PyTorch

# In[48]:


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[49]:


w = torch.ones(1)
b = torch.zeros(1)

t_p = model(t_u, w, b)
t_p


# In[50]:


loss = loss_fn(t_p, t_c)
loss


# ## La gradiente
# 
# **Como estimamos $w$ y $b$ para que el _loss_ llegue al minimo?**
# 
# * _Gradient descent_
#     * Calcular la razon de cambio del _loss_ con respecto a cada parametro
#     * Aplicar un cambio a cada parametro en la direccion que reduzca el _loss_

# In[51]:


delta = 0.1

loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c) -
                        loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)


# **Conceptualmente:** 
# * Un incremento unitario en $w$ va a provocar un cambio en el _loss_. 
#     * Si el cambio es negativo, tenemos que incrementar $w$ para minimizar el _loss_
#     * Si el cambio es positivo, tenemos que reducir $w$
#     
# **Por cuanto debemos de incrementar o reducir el valor de $w$?**
# * Proporcional a la razon de cambio del _loss_

# In[52]:


learning_rate = 1e-2

w = w - learning_rate * loss_rate_of_change_w


# In[53]:


loss_rate_of_change_b = (loss_fn(model(t_u, w, b + delta), t_c) -
                        loss_fn(model(t_u, w, b - delta), t_c) / (2.0 * delta))

b = b - learning_rate * loss_rate_of_change_b


# **Esto representa un paso en gradient descent**

# ## Version analitica
# 
# * Que pasa si el valor de delta fuera infinitesimamente pequenio?
# * La derivada del _loss_ con respecto a cada parametro.
#     * $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial t_p} \frac{\partial t_p}{\partial w}$

# In[54]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


# $\frac{\partial x^2}{\partial x} = 2x$

# In[55]:


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c)
    return dsq_diffs


# In[56]:


def model(t_u, w, b):
    return w * t_u + b


def dmodel_dw(t_u, w, b):
    return t_u


def dmodel_db(t_u, w, b):
    return 1.0


# La funcion retornando la gradiente del _loss_ con respecto a $w$ y $b$

# In[57]:


def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dw = dloss_fn(t_p, t_c) * dmodel_dw(t_u, w, b)
    dloss_db = dloss_fn(t_p, t_c) * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])


# ## Training loop
# 
# Ya tenemos todo para optimizar nuestros parametros:
# * Empezamos con un valor tentativo para cada paremetro
# * Actualizamos iterativamente
# * Paramos despues de un numero fijo de iteraciones
# * o hasta que $w$ y $b$ dejen de cambiar
# 
# 
# **Epoch**: una iteracion de entrenamiento durante la cual actualizamos los parametros para todo nuestro dataset de entrenamiento

# In[58]:


def training_loop(model, n_epochs, learning_rate, params, t_u, t_c, print_params=True):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        
        t_p = model(t_u, w, b) # Forward pass
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b) # Backward pass
        
        params = params - learning_rate * grad
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}, Loss {loss}")
            if print_params:
                print(f"\tParams: {params}")
                print(f"\tGrad: {grad}")
        
    return params


# In[59]:


training_loop(model,
              n_epochs = 100,
              learning_rate= 1e-2,
              params = torch.tensor([1.0, 0.0]),
              t_u = t_u,
              t_c = t_c)


# ### Que paso?
# 
# * Como podemos limitar la magnitud de `learning_rate * grad`?

# In[60]:


training_loop(model,
              n_epochs = 100,
              learning_rate= 1e-4,
              params = torch.tensor([1.0, 0.0]),
              t_u = t_u,
              t_c = t_c)


# ### Mejor
# 
# El comportamiento se mantuvo estable
# 
# Hay otro problema:
# * la gradiente

# ```
# Epoch 1, Loss 1763.8846435546875
# 	Params: tensor([ 0.5483, -0.0083])
# 	Grad: tensor([4517.2964,   82.6000])
# ```

# In[61]:


4517.2964/82.6000


# In[62]:


t_un = 0.1 * t_u


# In[63]:


training_loop(model,
              n_epochs = 100,
              learning_rate= 1e-2,
              params = torch.tensor([1.0, 0.0]),
              t_u = t_un, # normalizado
              t_c = t_c)


# In[64]:


params = training_loop(model,
                       n_epochs = 5000,
                       learning_rate= 1e-2,
                       params = torch.tensor([1.0, 0.0]),
                       t_u = t_un,
                       t_c = t_c,
                       print_params=False)

params


# * El _loss_ disminuyo mientras fuimos actualizando los parametros en la direccion de _gradient descent_
# * No llego a cero
#     * Porque?
#     
# * Sin embargo, los valores de $w$ y $b$ quedaron bastante cerca a los valores necesarios para convertir a Celsius a Fahrenheit.
# * Los valores exactos son:
#     * $w=5.5556$
#     * $b=-17.7778$

# In[65]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

t_p = model(t_un, *params)

fig = plt.figure(dpi=600)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')


# # PyTorch Autograd
# 
# * Back-propagation
#     * calcular la gradiente de una composicion de funciones - el modelo y el _loss_ - con respecto a sus parametros - $w$ y $b$ - propagando derivadas hacia atras usando la regla de la cadena.
# * Todas las funciones que deben ser diferenciables.
#     * Calcular la gradiente: la razon de cambio del _loss_ con respecto a los parametros
# 
# **Que pasa cuando tenemos un modelo con millones de parametros?**
# * Funcion diferenciable
# * calcular la gradiente
# * composicion de varias funciones lineales y no lineales

# In[66]:


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[67]:


params = torch.tensor([1.0, 0.0], requires_grad=True)


# `requires_grad=True`: Le estamos diciendo a PyTorch que mantenga un registro de todos los tensores resultantes de operaciones sobre `params`.
# 
# Cualquier tensor que tenga `params` como ancestro va a tener acceso a la cadena de funciones que se llamaron para llegar de `params` a ese tensor.
# 
# En caso que estas funciones sean diferenciables (la mayoria de operaciones en PyTorch lo son), el valor de la derivada va a ser automaticamente llenado como el atributo `grad` del tensor `params`.

# In[68]:


params.grad is None


# In[69]:


loss = loss_fn(model(t_u, *params), t_c)
loss.backward()

params.grad


# **dibujo**

# * Podemos tener $n$ tensores con `requires_grad=True`
# * y cualquier composicion de funciones
# * PyTorch calcularia las derivadas del _loss_ a traves de la cadena de esas funciones (grafica computacional)
# * **Acumularia** los valores en el atributo `grad` de esos tensores

# ## Warning
# 
# Llamar `backward` hace que las derivadas se **acumulen**. Tenemos que regresar la gradiente a cero explicitamente despues de usarla para actualizar parametros.

# In[70]:


if params.grad is not None:
    params.grad.zero_()


# In[71]:


def training_loop(model, n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()
            
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        # grad = grad_fn(t_u, t_c, t_p, w, b)
        loss.backward()
        
        params = (params - learning_rate * params.grad).detach().requires_grad_()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss {loss}")
            
    return params


# `params = (params - learning_rate * params.grad).detach().requires_grad_()`
# 
# Pensemos en la grafica computacional:
# * `p1 = (p0 * lr * p0.grad)`
#     * `p0` son los _weights_ del modelo
#     * `p0.grad` se calcula de una combinacion de `p0` y nuestra data a traves del _loss function_
# * `p2 = (p1 * lr * p1.grad)`
# * La grafica computacional de `p1` regresa a `p0` esto representa un problema:
#     * Tenemos que mantener `p0` en memoria hasta que termine el training
#     * Confunde donde tenemos que asignar el error a traves de back-prop
# * En vez, despegamos el nuevo tensor de `params` de la grafica computacional usando `detach()`
# * De esta forma `params` pierde la memoria de las operaciones que lo generaron.
# * Reanudamos el tracking llamando `requires_grad_()`
# * Ahora podemos deshacernos de la memoria mantenida por las versiones viejas de `params` y solo hacemos back-prop con los _weights_ actuales.

# In[72]:


training_loop(model=model,
              n_epochs=5000,
              learning_rate=1e-2,
              params=torch.tensor([1.0, 0.0], requires_grad=True), # CLAVE
              t_u = t_un, # Seguimos usando la version normalizada
              t_c = t_c)


# Ya no tenemos que calcular derivadas a mano :)

# ## Optimizadores
# 
# * Hemos estado utilizando gradient descent normal para optimizacion.
# * PyTorch abstrae la estrategia de optimizacion
# * El modulo `torch` tiene un submodulo `optim`

# In[73]:


import torch.optim as optim

dir(optim)


# Todo optmizador:
# * toma una lista de parametros (tensores de PyTorch, usualmente con `requires_grad=True`) y mantiene una referencia a ellos.
# * luego de que el _loss_ sea calculado con los inputs
# * una llamada a `.backward()` provoca que se llene `.grad` en los parametros
# * en ese punto, el optimizador puede accesar `.grad` para actualizar los parametros

# In[74]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5
optimizer = optim.SGD([params], lr=learning_rate)


# * SGD = Stochastic Gradient Descent
# * Exactamente lo mimsmo que hemos estado haciendo a mano (siempre y cuando el argumento `momentum` este en su valor default de 0.0)
# * El termino _stochastic_ viene del hecho que la gradiente normalmente se obitene promediando sobre un subset aleatorio de todos los inputs, llamado _minibatch_.

# In[75]:


t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
loss.backward()

optimizer.step()

params


# * El valor de `params` se actualizo cuando llamamos `step`
# * El optimizador utilizo los valores en `params.grad` y actualizo los parametros restando `(lr * grad)` de ellos.
# 
# **Que se nos olvido hacer arriba?**

# In[76]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)

optimizer.zero_grad()
loss.backward()
optimizer.step()

params


# In[77]:


def training_loop(model, n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss {loss}")
            
    return params


# In[78]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

training_loop(model,
              n_epochs=5000,
              optimizer=optimizer,
             params = params, # Es importante que ambos `params` sean el mismo objeto
             t_u = t_un,
             t_c = t_c)


# ## Adam

# In[79]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate) # Nuevo optimizador

training_loop(model,
              n_epochs=2000,
              optimizer=optimizer,
              params = params,
              t_u = t_u, # Regresamos a usar el t_u original como input
              t_c = t_c)


# ## En resumen
# 
# * Back-propagation para estimar la gradiente
# * autograd
# * optimizar los pesos del modelo usando la SGD u otros optimizadores

# # Ejercicios

# 1. Redefinan el model a `w2 * t_u ** 2 + w1 * t_u + b`
#     * Que partes del training loop necesitaron cambiar para acomodar el nuevo modelo?
#     * Que partes se mantuvieron iguales?
#     * El _loss_ resultante es mas alto o bajo despues de entrenamiento?
#     * El resultado es mejor o peor?

# In[80]:


def model(t_u, w1, w2, b):
    return w2 * t_u**2 + 2 + w1 * t_u + b


def loss_fn(t_p, t_c):
    diferencias_cuadradas = (t_p - t_c)**2
    return diferencias_cuadradas.mean()


# In[81]:


w1 = torch.ones(1)
w2 = torch.ones(1)
b = torch.zeros(1)

t_p = model(t_u, w1, w2, b)

loss = loss_fn(t_p, t_c)
loss


# In[82]:


delta = 0.1

learning_rate = 1e-2

loss_rate_of_change_b = (loss_fn(model(t_u, w1, w2, b + delta), t_c) -
                        loss_fn(model(t_u, w1, w2, b - delta), t_c) / (2.0 * delta))

b = b - learning_rate * loss_rate_of_change_b

loss_rate_of_change_w = (loss_fn(model(t_u, w1 + delta, w2 + delta, b), t_c) -
                        loss_fn(model(t_u, w1 - delta, w2 - delta, b), t_c)) / (2.0 * delta)


w = w - learning_rate * loss_rate_of_change_w


# # Descripción del proceso 
# 
# Las partes del training loop que tuvieron que adaptarse al nuevo modelo eran las demas funciones. Luego de modificar la función model() se añadieron las derivadas parciales de dicha función con respecto a las funciones w, w1 y w2. En  adición se calculó la función de perdida para los dos hiperparametros modificados en el ambito del loop de entrenamiento.
# 
# Las partes que se mantuvieron estables y sin moficiaciones son las que no implicaban la utilización de los hiperparametros w al contrario solo hacian uso de los parametros t. El calculo de la perdida en comparación a la variable real por ejemplo. 

# In[83]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()

def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c)
    return dsq_diffs

def dmodel_dw1(t_u, w, b):
    return t_u


def dmodel_dw2(t_u, w, b):
    return t_u**2


def dmodel_db(t_u, w1, w2, b):
    return 1.0

def grad_fn(w1, w2, t_u, t_c, t_p,  b):
    dloss_dw1 = dloss_fn(t_p, t_c) * dmodel_dw1(t_u, w1, b)
    dloss_dw2 = dloss_fn(t_p, t_c) * dmodel_dw2(t_u, w2, b)
    dloss_db = dloss_fn(t_p, t_c) * dmodel_db(t_u, w1, w2, b)
    return torch.stack([dloss_dw1.mean(), dloss_dw2.mean(), dloss_db.mean()])

def training_loop(model, n_epochs, learning_rate, params, t_u, t_c, print_params=True):
    for epoch in range(1, n_epochs + 1):
        w1,w2, b = params
        
        t_p = model(t_u, w1, w2, b) # Forward pass
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w1, w2, b) # Backward pass
        
        params = params - learning_rate * grad
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}, Loss {loss}")
            if print_params:
                print(f"\tParams: {params}")
                print(f"\tGrad: {grad}")
        
    return params


params = training_loop(model,
              n_epochs = 150,
              learning_rate= 1e-8,
              params = torch.tensor([1.0, 1.0, 0.0]),
              t_u = t_u,
              t_c = t_c)

params


# In[84]:




t_p = model(t_u, *params)

fig = plt.figure(dpi=600)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')


# In[85]:


t_un = 0.1 * t_u


# In[86]:


params = training_loop(model,
              n_epochs = 7000,
              learning_rate= 1e-10,
              params = torch.tensor([1, 0.2, 0.0]),
              t_u = t_un,
              t_c = t_c)

params


# In[87]:


t_p = model(t_u, *params)

fig = plt.figure(dpi=1000)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_u.numpy(), t_p.detach().numpy())


# In[88]:


get_ipython().system('jupyter nbconvert --to script Juarez_Boris_tarea3tensor-optimizations.ipynb')


# In[ ]:




