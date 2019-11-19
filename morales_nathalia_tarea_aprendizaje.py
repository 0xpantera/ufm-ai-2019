# Ejercicios

#1. Redefinan el model a `w2 * t_u ** 2 + w1 * t_u + b`
#    * Que partes del training loop necesitaron cambiar para acomodar el nuevo modelo?
# - la funcion del modelo se necesito cambiar
#    * Que partes se mantuvieron iguales?
# - la funcion de loss, y el training loop.
# - como la funcion del modelo cambio, se reconfiguraron los w1 y w2 para ser agregados a la lista de parametros
#    * El _loss_ resultante es mas alto o bajo despues de entrenamiento?
# - es igual a los ultimos tres entrenamientos
#    * El resultado es mejor o peor?
# - es igual pero utiliza una nueva manera de alcanzar los resultados dado el cambio del modelo


def model(w2, t_u, w1, b):
    return w2 * t_u ** 2 + w1 * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

def training_loop(model, n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss {loss}")

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # Temperatura en grados celsios
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # Unidades desconocidas
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

params = torch.tensor([1.0,1.0,0.0], requires_grad=True)
learning_rate= 1e-2
n_epochs = 5000
optimizer = optim.Adam([params], lr=learning_rate)

training_loop(model=model,
              n_epochs=n_epochs,
              params=params,
              optimizer = optimizer,
              t_u = t_u,
              t_c = t_c)

