# graph.py

class Graph:
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices[vertex] = []

    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.vertices and vertex2 in self.vertices:
            if vertex2 not in self.edges.get(vertex1, []):
                self.edges[vertex1] = self.edges.get(vertex1, []) + [vertex2]
            if vertex1 not in self.edges.get(vertex2, []):
                self.edges[vertex2] = self.edges.get(vertex2, []) + [vertex1]

    def remove_vertex(self, vertex):
        if vertex in self.vertices:
            del self.vertices[vertex]
            for adjacent_vertices in self.edges.values():
                if vertex in adjacent_vertices:
                    adjacent_vertices.remove(vertex)

    def remove_edge(self, vertex1, vertex2):
        if vertex1 in self.vertices and vertex2 in self.vertices:
            if vertex2 in self.edges.get(vertex1, []):
                self.edges[vertex1].remove(vertex2)
            if vertex1 in self.edges.get(vertex2, []):
                self.edges[vertex2].remove(vertex1)

    def get_adjacent_vertices(self, vertex):
        return self.edges.get(vertex, [])

    def get_vertices(self):
        return list(self.vertices.keys())

    def get_edges(self):
        return self.edges

    def is_connected(self):
        visited = set()
        stack = [next(iter(self.vertices))]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                stack.extend(self.get_adjacent_vertices(vertex))
        return len(visited) == len(self.vertices)

    def is_cyclic(self):
        visited = set()
        stack = [next(iter(self.vertices))]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                for adjacent_vertex in self.get_adjacent_vertices(vertex):
                    if adjacent_vertex in visited:
                        return True
                    stack.append(adjacent_vertex)
        return False
