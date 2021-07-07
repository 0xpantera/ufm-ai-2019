import torch
import matplotlib.pyplot as plt

# 1.) Crear un tensor de list(range(9)) e indicar cual es el size, offset, y strides
a = torch.tensor(range(9), dtype = torch.double)

# size: una tupla indicando cuantos elementos a traves de cada dimension representa el tensor
a.size()

# offset: indice en el storage correspondiente al primer elemento en el tensor
a.storage_offset()

#stride: numero de elementos en el storage que se necesitan saltar a para obtener el siguiente elemento a traves de cada dimension
a.stride()

#Crear un tensor b = a.view(3, 3). Cual es el valor de b[1, 1]
b = a.view(3,3)
print(b[1,1])

# crear un tensor c = b[1:, 1:]. Cual es el size, offset, strides?
c = b[1: , 1:]

# size: una tupla indicando cuantos elementos a traves de cada dimension representa el tensor
c.size()

# offset: indice en el storage correspondiente al primer elemento en el tensor.
c.storage_offset()

# stride: numero de elementos en el storage que se necesitan saltar para
#          obtener el siguiente elemento a traves de cada dimension.
c.stride()





# 2.) Escogan una operacion matematica como cosine o sqrt. Hay una funcion
#       correspondiente en PyTorch?
sqrtOP = torch.sqrt(a)

#Existe una version de esa operacion que opera in-place?
#R// Cualquier operación que mute un tensor in-place se escribe con un _ al final.
# Por ejemplo: x.copy_(y), x.t_(), cambiará x.
a.sqrt_()



# 3.) Crear un tensor 2D y luego agregar una dimension de tamaño 1 insertada en la dimension 0.
dim2 = torch.tensor([ [1, 2],
                     [3, 4] ])

# torch.unsqueeze(input, dim, out=None) → Tensor
#     Returns a new tensor with a dimension of size one inserted at the specified position.
dim2.unsqueeze_(0)
dim2.size()



# 4.) Eliminar la dimension extra que agrego en el tensor previo.
# torch.squeeze(input, dim=None, out=None) → Tensor
#    Returns a tensor with all the dimensions of input of size 1 removed.
dim2.squeeze_(0)
dim2.size()




# 5.) Crear un tensor aleatorio de forma  5𝑥3  en el intervalo  [3,7)
normal_tensor = torch.randn(5, 3, dtype=torch.double)


# 6.) Crear un tensor con valores de una distribucion normal (𝜇=0,𝜎=1)
import torch.distributions as tdist

# primera parametro es la media y el segundo la desviacion estandar
normDist = tdist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

# distribucion normal de 7 valores
normDist.sample((7,))



# 7.) Recuperar los indices de todos los elementos no cero en el tensor torch.Tensor([1,1,1,0,1]).
# obtiene los index del array donde los valores que son != de 0
import numpy as np
np.where(p != 0)

# obtiene los valores que son != de 0
p = torch.Tensor([1,1,1,0,1])
p[[p != 0]]


# 8.) Crear un tensor aleatorio de forma (3,1) y luego apilar cuatro copias horizontalmente.
al = torch.randn(3,1)

# 0: vertical, 1: horizontal
al = torch.cat((al.clone(),al.clone(),al.clone(),al.clone() ), 1)

al.size()

al = torch.randn(3,1)
al = torch.cat((al.clone(),al.clone(),al.clone(),al.clone() ), 0)

al.size()



# 9.) Retornar el producto batch matrix-matrix de dos matrices 3D:
#      (a=torch.randn(3,4,5), b=torch.rand(3,5,4))

a = torch.randn(3,4,5)
b = torch.rand(3,5,4)
torch.matmul(a, b)



# 10.) Retornar el producto batch matrix-matrix de una matriz 3D y una matriz 2D:
#       (a=torch.rand(3,4,5), b=torch.rand(5,4)).
a = torch.rand(3,4,5)
b = torch.rand(5,4)
torch.matmul(a, b)
