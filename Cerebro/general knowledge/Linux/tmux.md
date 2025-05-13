# Que es tmux?
tmux es un multiplexor de terminal que permite crear, gestionar y navegar entre múltiples sesiones de terminal dentro de una sola ventana. Es especialmente útil para trabajar en servidores remotos o para organizar múltiples tareas en una sola pantalla.

## Como funciona tmux
tmux tiene tres capas
1. Session
2. Window
3. Pane
## Cuales son los comandos más utilizados en tmux?

###  Iniciar sesión con tmux
`tmux`

### Detach sesión
`ctrl+b d`

### Volver a la sesión
`tmux a`

### Como ver las sesiones detach?
`tmux ls`

### Para crear una sesión con nombre
`tmux new -s <session_name>`
### Para ir a una sesión target de tmux?
`tmux a -t <session_name>` `

### Para terminar una sesión específica? 
`tmux kill-session -t <session_name>`

### Para hacer un nuevo pane
`ctrl+b %` (vertical)`
`ctrl+b "` (horizontal)


