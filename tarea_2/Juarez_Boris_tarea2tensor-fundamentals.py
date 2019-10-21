#!/usr/bin/env python
# coding: utf-8

# # Tensor Internals

# In[1]:


import torch


# Los _floating point numbers_ son la forma en la que las redes manejan la informacion.
# 
# La transformacion de un tipo de data a otro es normalmente aprendida por la red en etapas. 
# 
# Las representaciones son numeros que capturan la estructua de la data. 
# 
# Antes es importante entender como PyTorch maneja y guarda data - como input, representacion intermedia, y output.

# ### Tensores
# 
# La estructura de datos fundamental en PyTorch es el tensor. En este caso un tensor se refiere a una generalizacion de vectores y matrices a un numero arbitrario de dimensiones. Otro termino para el mismo son _arreglos multidimensionales_. La dimensionalidad de un tensor coincide con el numero de indices que se usan para referirse a un valor escalar dentro del tensor.

# In[2]:


zero_d = torch.tensor(1)
one_d = torch.tensor([1,2,3,4,5])

two_d = torch.tensor([[1,2,3],
                     [4,5,6]])

three_d = torch.ones((28,28,3))


# ## Comparacion con NumPy
# 
# PyTorch provee buena interoperabilidad con NumPy. Esto provee integracion de primera clase con el resto del stack cientifico de Python: `SciPy`, `Scikit-learn`, `Pandas`, `Matplotlib`, etc.
# 
# Es como NumPy en esteroides. Tiene la habilidad de realizar operaciones sumamente rapidas sobre GPUs, distribuir operaciones a traves de muchos dispositivos o maquinas y mantiene un registro de las computaciones realizadas.

# ## Objetivos
# 
# 1. Introducir tensores en PyTorch.
# 2. Manipular tensores en PyTorch.
# 3. Representacion de data en memoria.
# 4. Como se realizan operaciones en tensores de tamanios arbitrarios en tiempo constatne.
# 5. NumPy interoperability y aceleracion en GPUs.

# # Fundamentos del Tensor
# 
# * Un arreglo
# * Una estructua de datos que guarda una coleccion de numeros accesibles individualmente usando un indice.

# In[3]:


a = [1.0, 2.0, 1.0]


# In[4]:


a[0]


# In[5]:


a[2] = 3.0
a


# Usar listas para representar vectores de numeros como coordenadas de una linea es suboptimo:
# * Los numeros en Python son objetos.
# * Listas en Python estan hechas para colecciones sequenciales de objeos.
# * El interpretador de Python es lento comparado a codigo compilado optimizado.

# In[6]:


a = torch.ones(3)
a


# In[7]:


a[1]


# In[8]:


float(a[1])


# In[9]:


a[2] = 2.0
a


# Las listas en Python son colecciones de objetos individualmente alocados en memoria.
# 
# Los tensores de PyTorch o los arrays de NumPy son views (normalmente) sobre bloques contiguos de memoria que contienen tipos numericos de C _unboxed_, no objetos de Python - 32-bit (4 bytes) `float` en este caso (**NOTA** Agregar figura). Esto significa que un tensor de 1D de 1,000,000 de floats requiere exactamente 4,000,000 bytes continuos para ser almacenado en memoria, mas un pequenio overhead para meta data (e.i. dimensiones, tipo numerico)

# In[10]:


points = torch.FloatTensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points


# In[11]:


points[0, 1]


# In[12]:


points[0]


# El ultimo output es otro tensor de 1D de tamanio 2 que contiene los valores de la primera fila del tensor `points`. 
# 1. Se aloco otro chunk de memoria?
# 2. Se copiaron los valores a el?
# 3. Retorno la nueva memoria envuelta en un nuevo objeto `tensor`?

# ## Tensores y Almacenamiento
# 
# La implementacion debajo del capo:
# 
# 1. Los valores son alocados en bloques continuos de memoria, manejados por instancias de `torch.Storage`
# 2. Un `Storage` es un arreglo 1D de data numerica.
# 3. Un `Tensor` de PyTorch es una view sobre dicho `Storage` que es capaz de indexar dentro de ese storage usando un offset y pasos por dimension.
# 4. Varios tensores pueden indexar el mismo `Storage`.

# **AGREGAR DIBUJO**

# Podemos accesar el `storage` de un tensor usando la propiedad `.storage`

# In[13]:


points


# In[14]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()


# Aunque el tensor dice tener 3 filas y 2 columnas, en memoria es un arreglo continuo de tamanio 6. En este sentido el tensor, solo sabe como trasladar un par de indices a un espacio en memoria.
# 
# Podemos indexar la memoria de forma manual:

# In[15]:


points_storage = points.storage()
points_storage[0]


# No podemos indexar la memoria de un tensor 2D usando dos indices. El layout del almacenamiento siempre es 1D, a pesar de la dimensionalidad de cualquiera de los tensores que hagan referencia a el.
# 
# **IMPORTANTE** Cambiar el valor de un almacenamiento cambia el contenido de los tensores que le hacen referencia:

# In[16]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storage = points.storage()
points_storage[0] = 2.0
points


# ## Size, offset, stride
# 
# Para poder indexar a un `storage`, los tensores hacen uso de la siguiente informacion: size, offset y stride.
# 
# * size: una tupla indicando cuantos elementos a traves de cada dimension representa el tensor.
# * offset: indice en el storage correspondiente al primer elemento en el tensor.
# * stride: numero de elementos en el storage que se necesitan saltar para obtener el siguiente elemento a traves de cada dimension.

# **AGREGAR DIBUJO**

# In[17]:


points.shape


# In[18]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point.storage_offset()
second_point


# In[19]:


second_point.size()


# In[20]:


points.stride()


# In[21]:


points.stride()


# * El tensor resultante tiene un offset de 2 en el storage.
# * size es una instancia de la clase `Size` que contiene un elemento, ya que el tensor es de 1D.
# * stride es una tupla que indica el numero de elementos en el storage que tienen que ser saltados cuando el indice es incrementado por 1 en cada dimension.
# 
# Accesar el elemento $i, j$ en un tensor 2D resulta en accesar el elemento `storage_offset + stride[0] * i + stride[1] * j` en el storage.

# La relacion entre `Tensor` y `Storage` hace que algunas operaciones, como transponer un tensor o extraer un sub-tensor, sean baratas ya que no requieren realocar memoria. En vez consisten en alocar una nueva instancia del objeto tensor con un diferente valor de size, storage, stride.

# In[22]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point.size()


# In[23]:


second_point.storage_offset()


# In[24]:


second_point.stride()


# El sub-tensor tiene una dimension menos mientras sigue indexando el mismo storage que el tensor original. Esto tambien significa que modificar el sub-tensor va a tener un efecto en el original:

# In[25]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point
second_point[0] = 10.0


# In[26]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point[0] = 10.0
points


# In[27]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1].clone()
second_point[0] = 10.0
points


# Transpuesta: modificar el tensor que tiene puntos individuales en las filas y coordenadas $x,y$ en las columnas y voltearlo para que los puntos individuales esten a lo largo de las columnas

# In[28]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points


# In[29]:


points_t = points.t()
points_t


# In[30]:


id(points.storage()) == id(points_t.storage())


# In[31]:


points.stride()


# In[32]:


points_t.stride()


# In[33]:


x


# In[34]:


x = torch.ones((3, 4, 5))
x_t = x.transpose(0, 2)
x_t


# In[35]:


x_t.size()


# In[36]:


x.stride()


# In[37]:


x_t.stride()


# Un tensor cuyos valores estan representados en el storage empezando desde la ultima dimension en adelante (e.i. moviendose a lo largo de las filas para un tensor 2D) se define como `continuo`. Los tensores `continuos` son convenientes porque podemos visitarlos en orden de forma eficiente sin saltar en el storage.

# In[38]:


x.is_contiguous()


# In[39]:


x_t.is_contiguous()


# ## Tipos Numericos

# * Que tipos numericos podemos almacenar en un `Tensor`?
# * El argumento `dtype` a los constructores del tensor (funciones como `tensor`, `zeros`, `ones`) especifica el tipo de data numerico que va a almacenar el tensor.
# * El tipo de dato especifica los posibles valores del tensor (enteros vs. floating point numbers) y el numero de bytes por valor.

# ### Posibles tipos de datos

# * `torch.float32` o `torch.float`: 32-bit floating point
# * `torch.float64` o `torch.double`: 64-bit double precision floating point
# * `torch.float16` o `torch.half`: 16-bit, half precision floating point
# * `torch.int8`: signed 8-bit integers
# * `torch.uint8`: unsigned 8-bit integers
# * `torch.int16` o `torch.short`: signed 16-bit integers
# * `torch.int32` o `torch.int`: signed 32-bit integers
# * `torch.int64` o `torch.long`: signed 65-bit integers

# **Task:**
# * Cual es el rango de valores que cada uno de estos tipos puede representar?
# * Cual es la diferencia entre un `int` y un `uint`?
# * Cual es el tipo default que utiliza base python? NumPy? PyTorch?

# In[40]:


double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)


# In[41]:


short_points.dtype


# In[42]:


double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()


# In[43]:


double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)


# Under the hood ambos metodos (`type` y `to`) realizan el la misma revision y conversion si es necesaria. Pero el metodo `to` toma argumentos adicionales.

# Podemos hacer cast de un tensor de un tipo a un tensor de otro tipo utilizando el metodo `type`

# In[44]:


# randn inicializa los elementos del tensor a numeros aleatorios entre 0 y 1
points = torch.randn(10, 2)
short_points = points.type(torch.short)


# ## Indexando Tensores

# Vimos que `points[0]` retorna un tensor que contiene el punto 2D en la primera fila del tensor.
# 
# Que pasa si queremos obtener un tensor que contenga todos los puntos excepto el primero?

# In[45]:


some_list = list(range(6))
some_list[:] # Todos los elementos de la lista
some_list[1:4] # Del elemento 1 inclusivo hasta el 4 exclusivo
some_list[1:] #  Del elemento 1 inclusivo hasta el final de la lista
some_list[:4] # Del primer elemento (indice 0) al 4 exclusivo
some_list[:-1] # del principio de la lista al ante penultimo
some_list[1:4:2] # Del elemento 1 inclusivo al elemento 4 exclusivo en pasos de 2


# Podemos hacer lo mismo en PyTorch con el beneficio adicional de poder utilizar range indexes para cada dimension del tensor.

# In[46]:


points


# In[47]:


points[1:]  # Todas las filas despues de la primera, implicitamente todas las columnas
points[1:, :] # Todas las filas despues de la primera, todas las columnas
points[1:, 0] # Todas las filas despues de la primera, la primera columna


# In[48]:


points[1:, 0]


# ## Interop con NumPy
# 
# Los tensores de PyTorch pueden ser convertidos a arrays de NumPy y vice versa de forma eficiente. Dada la ubiquidad de NumPy en el ecosistema de data science de Python esto es algo importante. Esto nos permite aprovechar una enorme cantidad de funcionalidad que el ecosistema de Python a desarrollado alrededor del NumPy Array.

# In[49]:


points = torch.ones(3, 4)
points_np = points.numpy()
type(points_np)


# In[50]:


points = torch.from_numpy(points_np)
points


# ## Serializando tensores
# 
# Crear tensores de forma dinamica esta bien, pero si la data dentro de los tensores es valiosa, vamos a querer guardarla a un archivo y cargarla de nuevo en algun punto.
# 
# PyTorch utiliza `pickle` para serializar tensores

# In[51]:


torch.save(points, "data/serialization-example/our_points.t")


# In[52]:


with open("data/serialization-example/our_points.t", "wb") as f:
    torch.save(points, f)


# In[53]:


points = torch.load("data/serialization-example/our_points.t")


# In[54]:


with open("data/serialization-example/our_points.t", "rb") as f:
    points = torch.load(f)


# In[55]:


import h5py

f = h5py.File("data/serialization-example/our_points.hdf5", "w")
dset = f.create_dataset("coords", data=points.numpy())
f.close()


# * "coords" es una llave en el archivo HDF5. Podemos tener otras llevas e incluso llaves anidadas. 
# * HDF5 nos permite indexar el dataset mientras esta en el disco y solo acceder los elementos que nos interesan.
# 
# Por ejemplo: Podemos cargar solo los ultimos dos puntos de nuestro dataset.

# In[56]:


fs = h5py.File("data/serialization-example/our_points.hdf5", "r")
dset = fs["coords"]

last_points = dset[1:]
last_points


# En este caso la data no fue cargada cuando el archivo se abrio o el dataset fue requerido. En vez, la data se mantuvo en disco hasta que solicitamos las ultimas dos filas. 
# 
# En ese punto, `h5py` acceso regreso un objeo similar a un array de NumPy que encapsula esa region del dataset y se comporta igual que un array de NumPy y tiene el mismo API. 

# In[57]:


last_points = torch.from_numpy(last_points)
f.close()


# In[58]:


last_points


# ## Moviendo tensores al GPU
# 
# Todos los tensores `Torch` pueden ser transferidos a un GPU para realizar computaciones en paralelo. Todas las operaciones que van a ser realizadas sobre el tensor seran llevadas a cabo usando rutinas especificas al GPU que vienen con PyTorch.

# Adicional al `dtype`, un `Tensor` de PyTorch tambien tiene el argumento de `device`, que establece donde en la computadora se va a poner la data del tensor.

# In[ ]:


import torch


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]]).to(device)


# In[6]:


points = 2 * points
points_gpu = 2 * points.to(device)


# El tensor `points_gpu` no regrese al CPU una vez los resultados han sido calculados. Lo que paso arriba es lo siguiente:
# 
# 1. El tensor `points` fue copiado al GPU
# 2. Un nuevo tensor fue alocado en el GPU y usado para guardar los resultados de la multiplicacion
# 3. Un handle al tensor GPU fue retornado

# In[44]:


points_gpu = points_gpu + 4


# Si agregamos un constante al resultado, la suma fue realizada en el GPU, la informacion no fluye al CPU (excepto si intentamos acceder el tensor ei. `print()`). Para mover el tensor de vuelta al CPU necesitamos proveer el argumento `cpu` al metodo `to`.

# In[45]:


points_cpu = points_gpu.to(device='cpu')


# ## El Tensor API

# [PyTorch Docs](http://pytorch.org/docs)

# La mayoria de operaciones sobre y entre tensores estan disponibles bajo el modulo `torch` y pueden ser llamadas como metodos del objeto `tensor`.
# 
# La operaciones sobre tensores estan divididas en grupos:
# 
# * Creation ops: funciones para construir un tensor, como `ones` y `from_numpy`
# * Indexing, slicing, joining, mutating ops: funciones para cambiar el `shape`, `strid` o los contenidos de un tensor, como `transpose`
# * Math ops: funciones para manipular el contenido de un tensor a traves de computaciones:
#     * Pointwise ops: funciones para obtener un nuevo tensor al aplicar una funcion a cada elemento de forma independiente. `abs` y `cos`
#     * Reduction ops: funciones para calcular valores agregados al iterar sobre tensores. `mean`, `std` y `norm`
#     * Comparison ops: funciones para evaluar predicados numericos sobre tensores. `equal` y `max`
#     * Spectral ops: funciones para transformar y operar en frequencias. `stft` y `hamming_window`
#     * Other ops: funciones especiales para operar sobre vectores, como `cross` o matrices como `trace`
#     * BLAS y LAPACK ops: funciones que siguen las especificaciones BLAS (Basic Linear Algebra Subprograms) para operaciones escalares, vector-vector, matrix-vector, matrix-matrix
#     * Random sampling: funciones para generar valores muestreando aleatoriamente de distribuciones de probabilidades. `randn` y `normal`
#     * Serialization: funciones para guardar y cargar tensores como `load` y `save`
#     * Parallelism: funciones para controlar el numero de threads para ejecucion paralela en el CPU, como `set_num_threads`

# ## Ejercicios

# ## 1
# 
# a. Crear un tensor de `list(range(9))` e indicar cual es el `size`, `offset`, y `strides`
# 
# b. Crear un tensor `b = a.view(3, 3)`. Cual es el valor de `b[1, 1]`
# 
# c. crear un tensor `c = b[1:, 1:]`. Cual es el `size`, `offset`, `strides`?

# In[62]:


# a
a = torch.tensor(list(range(9)))
print(a)
print("Size: ", a.size())
print("Offset: ",a.storage_offset())
print("Stride: ",a.stride())


# In[64]:


# b
b = torch.tensor(a.view(3, 3))
print(b[1,1])
print("TENSOR Primer ejercicio inciso B: ", b, "\n")


# In[65]:


# c 
c = b[1:, 1:]
print("TENSOR Primer ejercicio inciso C: ", c, "\n")
print("Size: ", c.size())
print("Offset: ", c.storage_offset())
print("Stride: ", c.stride())


# ## 2
# ### Escogan una operacion matematica como cosine o sqrt. 
# 
# ### A. Hay una funcion correspondiente en PyTorch?
# 
# Si existe y su sintaxis es torch.sqrt
# 
# ### B. Existe una version de esa operacion que opera `in-place`?
# 
# y=torch.sqrt(x) devuelve un nuevo tensor con la raíz cuadrada de los elementos del tensor x  
# 
# x:sqrt() reemplaza todos los elementos presentes en el tensor con la raíz cuadrada de los elementos del tensor x
# 

# # Ejercicios 
# ## 1. Crear un tensor 2D y luego agregar una dimension de tamanio 1 insertada en la dimension 0.

# In[67]:


#Ejercicios 1 
tensor_de_ejercicios = torch.tensor([[1.0], [2.0]])
print("Tensor 2D previo: ", tensor_de_ejercicios)

tensor_agregado = torch.zeros(1, 1)
tensor_de_ejercicios = torch.cat((tensor_agregado, tensor_de_ejercicios), 0)
print("Tensor 2D despues: ", tensor_de_ejercicios)


# ## 2. Eliminar la dimension extra que agrego en el tensor previo.

# In[84]:


#Ejercicios 2
tensor_de_ejercicios = tensor_de_ejercicios[1:,:]
print(tensor_de_ejercicios)


# ## 3. Crear un tensor aleatorio de forma $5x3$ en el intervalo $[3,7)$

# In[85]:


# Ejercicios 3 

tensor_aleatorio = torch.randint(3, 7, (5, 3))
print(tensor_aleatorio)


# ## 4. Crear un tensor con valores de una distribucion normal ($\mu=0, \sigma=1$)

# In[86]:


#Ejercicios 4
tensor_normdist = torch.randn(2, 4, dtype=torch.double)
print(tensor_normdist)


# ## 5. Recuperar los indices de todos los elementos no cero en el tensor `torch.Tensor([1,1,1,0,1])`.

# In[87]:


# Ejercicios 5
Tensor_5 = torch.Tensor([1,1,1,0,1])
index_nonzero = torch.nonzero(Tensor_5)
print(index_nonzero)


# ## 6. Crear un tensor aleatorio de forma `(3,1)` y luego apilar cuatro copias horizontalmente.

# In[88]:


#Ejercicios 6
tensor_6 = torch.rand(3, 1)
tensor_apilado = torch.cat((tensor_6, tensor_6, tensor_6, tensor_6), 1)
print("Tensor ORIGINAL: ",tensor_6)
print("Tensor Apilado: ", tensor_apilado)


# ## 7. Retornar el producto batch matrix-matrix de dos matrices 3D: (`a=torch.randn(3,4,5)`, `b=torch.rand(3,5,4)`)

# In[89]:


# Ejercicios 7

primer_tensor=torch.randn(3,4,5)
segundo_tensor=torch.rand(3,5,4)
torch.bmm(primer_tensor,segundo_tensor)


# ## 8. Retornar el producto batch matrix-matrix de una matriz 3D y una matriz 2D: (`a=torch.rand(3,4,5)`, `b=torch.rand(5,4)`).

# In[90]:


# Ejercicios 8

primer_tensor=torch.rand(3,4,5)
segundo_tensor=torch.rand(5,4)
torch.matmul(primer_tensor, segundo_tensor)


# In[91]:


pip install nbconvert


# In[92]:


get_ipython().system('jupyter nbconvert --to script config_template.ipynb')


# In[ ]:




