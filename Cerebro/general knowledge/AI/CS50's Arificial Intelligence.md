<iframe width="560" height="315" src="https://www.youtube.com/embed/5NgNicANyqM?si=uK-D0lXC58XYgQJ5" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
# Curso de Harvard de AI
___
## Conceptos:
- [x] Search ✅ 2023-11-27
- [ ] Knowledge
- [ ] Uncertainty
- [ ] Optimization
- [ ] Learning
- [ ] Neural Networks
- [ ] Language

## Search:
Los problemas de busquedas son como los laberintos, buscar el camino más corto en un mapa etc.

### Agent
> Entity that perceives its environment and acts upon that environment 

En el caso de las direcciones por ejemplo el agente puede ser una representación del coche que esta tratando de saber que acciones tiene que hacer para llegar a su destino.

### Search problems
1. Initial state
2. Actions
3. Transition model
4. Goal test
5. Path cost function
### State
> A configuration of the agent and its environment

Es el estado, la configuración del escenario.

### Actions
> Choices that can be made in a state

ACTIONS(_s_) returns the set of actions that can be executed in state _s_ 

### Transition model
> A description of what state results from performing any applicable action in any state

RESULTS(_s, a_) returns the state resulting from performing action _a_ in state _s_ 

### State space
> The set of all states reachable from the initial state by any sequence of actions

### Goal test
> Way to determine whether a given state is a goal state

### Path cost
> Numerical cost associated with a given path

### Solution
> A sequence of actions that leads from the initial state to a goal state

### Optimal solution
> A solution that has the lowest path cost among all solutions

### Node
> A data structure that keeps track of
> - a state
> - a parent (node that generated this node)
> - an action (action applied to parent to get node)
> - a path cost (from initial state to node)

### Algoritmos de busquedas:
#### Depth-first search
> Search algorithm that always expands the deepest node in the frontier. Usa un STACK

#### Breadth-first search
> Search algorithm that always expands the shallowest node in the frontier. Usa un QUEUE

#### Uninformed search
> Search strategy that uses no problem-specific knowledge

#### Informed search
> Search strategy that uses problem-specific knowledge to find solutions more efficiently

#### Greedy best-first search
> Search algorithm that expands the node that is closet to the goal, as estimated by a heuristic function 

#### A* search
> Search algorithm that expands node with lowest value of g(n) + h(n)
> g(n) = cost to reach node
> h(n) = estimated cost to goal

#### Minimax algorithm

#### Implementación:
```python
import sys

class Node():
	def __init__(self, state, parent, action):
		self.state = state
		self.parent = parent
		self.action = action


class StackFrontier():
	def __init__(self):
		self.frontier = []

	def add(self, node):
		self.frontier.append(node)

	def contains_state(self, state):
		return any(node.state == state for node in self.frontier)

	def empty(self):
		return len(self.frontier) == 0

	def remove(self):
		if self.empty():
			raise Exception("empty frontier")
		else:
			node = self.frontier[-1]
			self.frontier = self.frontier[:-1]
			return node


class QueueFrontier(StackFrontier):

	def remove(self):
		if self.empty():
			raise Exception("empty frontier")
		else:
			node = self.frontier[0]
			self.frontier = self.frontier[1:]
			return node


class Maze():

	def __init__(self, filename):
		# Read file and set height and width of maze
		with open(filename) as f:
			contents = f.read()

		# Validate start and goal
		if contents.count("A") != 1:
			raise Exception("maze must have exactly one start point")
		if contents.count("B") != 1:
			raise Exception("maze must have exactly one goal")
		# Determine height and width of maze
		contents = contents.splitlines()
		self.height = len(contents)
		self.width = max(len(line) for line in contents)

		# Keep track of walls
		self.walls = []
		for i in range(self.height):
			row = []
			for j in range(self.width):
				try:
					if contents[i][j] == "A":
						self.start = (i, j)
						row.append(False)
					elif contents[i][j] == "B":
						self.goal = (i, j)
						row.append(False)
					elif contents[i][j] == " ":
						row.append(False)
					else:
						row.append(True)
				except IndexError:
					row.append(False)
			self.walls.append(row)
		self.solution = None


	def print(self):
		solution = self.solution[1] if self.solution is not None else None
		print()
		for i, row in enumerate(self.walls):
			for j, col in enumerate(row):
				if col:
					print(" ", end="")
				elif (i, j) == self.start:
					print("A", end="")
				elif (i, j) == self.goal:
					print("B", end="")
				elif solution is not None and (i, j) in solution:
					print("*", end="")
				else:
					print(" ", end="")
			print()
		print()


	def neighbors(self, state):
		row, col = state

		# All possible actions
		candidates = [
			("up", (row - 1, col)),
			("down", (row + 1, col)),
			("left", (row, col - 1)),
			("right", (row, col + 1))
		]

		# Ensure actions are valid
		result = []
		for action, (r, c) in candidates:
			try:
				if not self.walls[r][c]:
					result.append((action, (r, c)))
			except IndexError:
				continue
		return result


	def solve(self):
		"""Finds a solution to maze, if one exists."""

		# Keep track of number of states explored
		self.num_explored = 0

		# Initialize frontier to just the starting position
		start = Node(state=self.start, parent=None, action=None)
		frontier = StackFrontier()
		frontier.add(start)

		# Initialize an empty explored set
		self.explored = set()

		# Keep looping until solution found
		while True:
			#If nothing left in frontier, then no path
			if frontier.empty():
				raise Exception("no solution")

			# Choose a node from the frontier
			node = frontier.remove()
			self.num_explored += 1

			# If node is the goal, then we have a solution
			if node.state == self.goal:
				actions = []
				cells = []

				# Follow parent nodes to find solution
				while node.parent is not None:
					actions.append(node.action)
					cells.append(node.state)
					node = node.parent
				actions.reverse()
				cells.reverse()
				self.solution = (actions, cells)
				return

			# If self.state is not the goal
			# Mark node as explored
			self.explored.add(node.state)

			# Add neighbors to frontier
			for actions, state in self.neighbors(node.state):
				if not frontier.contains_state(state) and state not in self.explored:
					child = Node(state=state, parent=node, action=action)
					frontier.add(child)

# Imcompleto !!!
			
```

## Knowledge
