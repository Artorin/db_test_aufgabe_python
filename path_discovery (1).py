"""A graph path discovery coding task.

In this file, you are presented with the task to implement the function `compute_shortest_paths`
which discovers some shortest paths from a start node to an end node in an undirected weighted graph
with strictly positive edge lengths.
"""
from functools import total_ordering
from typing import Any, List, Optional, List, Tuple, cast

import copy


class Node:
    """A node in a graph."""

    def __init__(self, id: int):
        self.id: int = id
        self.adjacent_edges: List["UndirectedEdge"] = []

    def edge_to(self, other: "Node") -> Optional["UndirectedEdge"]:
        """Returns the edge between the current node and the given one (if existing)."""
        matches = [edge for edge in self.adjacent_edges if edge.other_end(self) == other]
        return matches[0] if len(matches) > 0 else None

    def is_adjacent(self, other: "Node") -> bool:
        """Returns whether there is an edge between the current node and the given one."""
        return other in {edge.other_end(self) for edge in self.adjacent_edges}

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.id == other.id

    def __le__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.id <= other.id

    def __hash__(self) -> int:
        return self.id

    def __repr__(self) -> str:
        return f"Node({self.id})"


class UndirectedEdge:
    """An undirected edge in a graph."""

    def __init__(self, end_nodes: Tuple[Node, Node], length: float):
        self.end_nodes: Tuple[Node, Node] = end_nodes
        if 0 < length:
            self.length: float = length
        else:
            raise ValueError(
                f"Edge connecting {end_nodes[0].id} and {end_nodes[1].id}: "
                f"Non-positive length {length} not supported."
            )

        if any(e.other_end(end_nodes[0]) == end_nodes[1] for e in end_nodes[0].adjacent_edges):
            raise ValueError("Duplicate edges are not supported")

        self.end_nodes[0].adjacent_edges.append(self)
        if self.end_nodes[0] != self.end_nodes[1]:
            self.end_nodes[1].adjacent_edges.append(self)
        self.end_node_set = set(self.end_nodes)

    def other_end(self, start: Node) -> Node:
        """Returns the other end of the edge, given one of the end nodes."""
        return self.end_nodes[0] if self.end_nodes[1] == start else self.end_nodes[1]

    def is_adjacent(self, other_edge: "UndirectedEdge") -> bool:
        """Returns whether the current edge shares an end node with the given edge."""
        return len(self.end_node_set.intersection(other_edge.end_node_set)) > 0

    def __repr__(self) -> str:
        return (
            f"UndirectonalEdge(({self.end_nodes[0].__repr__()}, "
            f"{self.end_nodes[1].__repr__()}), {self.length})"
        )



class UndirectedGraph:
    """A simple undirected graph with edges attributed with their length."""

    def __init__(self, edges: List[UndirectedEdge]):
        self.edges: List[UndirectedEdge] = edges
        self.nodes_by_id = {node.id: node for edge in self.edges for node in edge.end_nodes}


@total_ordering
class UndirectedPath:
    """An undirected path through a given graph."""

    def __init__(self, nodes: List[Node]):
        assert all(
            node_1.is_adjacent(node_2) for node_1, node_2 in zip(nodes[:-1], nodes[1:]) #  1,2 and 2,1:  1.2 , 2.1
        ), "Path edges must be a chain of adjacent nodes"
        self.nodes: List[Node] = nodes
        self.length = sum(
            cast(UndirectedEdge, node_1.edge_to(node_2)).length
            for node_1, node_2 in zip(nodes[:-1], nodes[1:])
        )

    @property
    def start(self) -> Node:
        return self.nodes[0]

    @property
    def end(self) -> Node:
        return self.nodes[-1]

    def prepend(self, edge: UndirectedEdge) -> "UndirectedPath":
        if self.start not in edge.end_nodes:
            raise ValueError("Edge is not adjacent")
        return UndirectedPath([edge.other_end(self.start)] + self.nodes)

    def append(self, edge: UndirectedEdge) -> "UndirectedPath":
        if self.end not in edge.end_nodes:
            raise ValueError("Edge is not adjacent")
        return UndirectedPath(self.nodes + [edge.other_end(self.end)])

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, UndirectedPath) and self.nodes == other.nodes

    def __le__(self, other: Any) -> bool:
        return isinstance(other, UndirectedPath) and self.length <= other.length

    def __hash__(self) -> int:
        return hash(n.id for n in self.nodes)

    def __repr__(self) -> str:
        nodestr: str = ", ".join([node.__repr__() for node in self.nodes])
        return f"UndirectedPath([{nodestr}])"


def compute_shortest_paths(
    graph: UndirectedGraph, start: Node, end: Node, length_tolerance_factor: float) -> List[UndirectedPath]:

    # graph anders vorstellen
    changed_graph = change_struct_graph(graph,len(graph.nodes_by_id))

    # start , ende
    source = str(start.id)
    sink = str(end.id)

    # zuerst den kürzeste Pfad finden
    path, distance = dijkstra(changed_graph, source, sink)
    res1 = tuple(zip([path], [distance]))

    # max length der anderen potenziellen Pfaden
    max_len = length_tolerance_factor * distance
    # Pfaden mit Wiederholungen finden
    res2 = find_another(changed_graph, source, sink, max_len, path, distance)
    # Pfaden die in der Nähe von dijkstra Pfad
    res3 = find_in_dijkstra(changed_graph, path, source, sink, max_len)

    founded_paths = []

    # alles sammeln
    for i in res1:
        founded_paths.append(i[0])
    for i in res2:
        founded_paths.append(i[0])
    for i in res3:
        founded_paths.append(i[0])

    # nodes sammeln
    nodes_list = []
    for i in range(len(founded_paths)):
        nodes: List[Node] = []
        nodes_list.append(nodes)

    # nodes in die Liste ablegen
    counter = 0
    for i in founded_paths:
        for j in i:
            node = graph.nodes_by_id[int(j)]
            nodes_list[counter].append(node)
        counter += 1

    # UndirectedPaths rekonstruieren
    counter = 0
    upaths_list: List[UndirectedPath] = []
    for i in range(len(founded_paths)):
        upath = UndirectedPath(nodes_list[counter])
        upaths_list.append(upath)
        counter += 1
    return upaths_list

# konzentrieren nur auf nodes, for the sake of simplicity, (in der Aufgabe stand nicht dass man die Lösung auf Kanten setzten soll (: )
def change_struct_graph(demo_graph, number):
    obj = {}

    # alle Kanten
    i = 0
    while i < number:
        i += 1
        obj['' + str(i)] = []

    print(obj)

    #Representation von Kanten umbauen
    # am Ende sieht aus wie:
    # {'1': [{'2': 10}, {'3': 30}], '2': [{'1': 10}, {'4': 10}], '3': [{'1': 30}, {'4': 10}], '4': [{'2': 10}, {'3': 10}]}
    for i in demo_graph.edges:  # UndirectedEdge((n1, n2), 10)

        n1 = i.end_nodes[0].id  # n1.id
        n2 = i.end_nodes[1].id  # n2.id
        length = i.length  # 10

        obj[str(n1)].append({"" + str(n2): length})  # to n1 {n2:10}
        obj[str(n2)].append({"" + str(n1): length})

    print(obj)
    # dicts mergen
    for i in obj:
        res = {}
        len_of = len(obj[str(i)])
        for j in range(len_of):
            res.update(obj[str(i)][j])

        obj[str(i)].clear()
        obj[str(i)] = dict(res)

    return obj

# wird nicht benutzt, weil findet n Pfaden, aber solche die nicht an tolerance wert passen
def YenKSP(graph, source, sink, k_paths):
    path, distances = dijkstra(graph, source, sink)
    A = [path]  # [1,2,4]
    upper = 50
    dijkstra_cost = distances  # 20
    B = []

    class Map(dict):
        def __getattr__(self, k):
            return self[k]
        def __setattr__(self, k, v):
            self[k] = v

    # todo find all another paths , what less then X*distances from dijkstra
    # while spur_cost < upper + 5:

    for _ in range(1, k_paths):
        for i in range(len(A[-1]) - 1):  # A[-1] : ["1","2"]
            print("i")
            print(i)
            graph_clone = copy.deepcopy(graph)

            spur_node = A[-1][i]  # A[-1][0] : "1"
            root_path = A[-1][:i + 1]  # [1,2]
            print("root_path")
            print(root_path)

            print("for path in A")
            for path in A:
                print(path)
            for path in A:
                if len(path) >= i and root_path == path[:i + 1]:
                    if path[i + 1] in graph_clone[path[i]]:
                        print("graph_clone[path[i]]")
                        print(graph_clone[path[i]])

                        graph_clone[path[i]].pop(path[i + 1])

                        graph_clone[path[i + 1]].pop(path[i])

            # todo
            # for each node rootPathNode in rootPath except spurNode:
            # remove rootPathNode from Graph;

            spur_path, spur_cost = dijkstra(graph_clone, spur_node, sink)

            if spur_path == None:
                spur_path = []

            total_path = root_path[:-1] + spur_path

            B.append(Map({'path': total_path, 'cost': spur_cost}))
            print("spur cost")
            print(spur_cost)
            if spur_cost >= upper:
                break

        if len(B) == 0:
            break

        B.sort(key=lambda p: (p.cost, len(p.path)))
        best_b = B.pop(0)
        if best_b.cost != float('inf'):
            A.append(best_b.path)

    return A



# greedy algorithm, man könnte aber A* algorithm nutzen, allerdings müsste man extra Koordinaten setzen um heuristic zu bekommen
def dijkstra(graph, source, target):
    distances = {node: float('inf') for node in graph} # alle nicht erreichbar vom start
    distances[source] = 0 # Die Distanz des Startknotens (source) wird auf 0 gesetzt
    previous = {} # Vorgängerknoten speichern
    unvisited = set(graph.keys()) # alle sind unvisited nodes

    while unvisited: # solange es noch unbesuchte Knoten gibt
        current_node = min(unvisited, key=lambda node: distances[node]) # der Knoten mit der kleinsten Distanz
        unvisited.remove(current_node) # besucht

        if current_node == target: # trifft endknote
            break

        for neighbor, weight in graph[current_node].items(): #alle Nachbarknoten des current_node besuchen
            new_distance = distances[current_node] + weight # Distanz berechnen
            if new_distance < distances[neighbor]: # neue Distanz kleiner ist als die aktuelle Distanz des Nachbarn
                distances[neighbor] = new_distance # Distanz für den Nachbarn aktualisieren
                previous[neighbor] = current_node # weiter gehen

    if target not in previous: # ob der Zielknoten (target) einen Vorgängerknoten hat - true - keinen Pfad zum Ziel gibt
        return None, float('inf')

    #rekonstruieren
    path = []
    current_node = target
    while current_node != source:
        path.append(current_node)
        current_node = previous[current_node]
    path.append(source)
    path.reverse()

    return path, distances[target]


# solve partially the problem with repeats and finds the paths that repeats near from the shortest path
def find_another(graph, source, sink, upper, path, distances):
    visited_nodes = [path[-1]]
    founded_paths = []
    founded_lengths = []

    def find_reversed(k, v, path_now, len_now, upper, visited_nodes, sink, founded_paths, new_path):
        # dijkstra pfad len und path
        len_dijkstra_start = len_now
        path_d_unchange = path_now

        # TODO for k1, v1 in graph[k].items():  # damit aus dem Node auf alle Pfaden gehen - am Debugging noch
        #len of Path now, upper - max length of path
        if len_now + v <= upper:

            if k not in visited_nodes:  # without repeats
                if len_now <= upper:  # check if can go back
                    visited_nodes.append(k)

                    if k != sink:  # is start
                        # build new path
                        new_path += [k]
                        len_now += v
                        # Pfad bauen
                        if (len_now - len_dijkstra_start) * 2 + len_dijkstra_start <= upper:
                            founded_paths.append(path_now + new_path[:-1] + new_path[::-1] + [sink])
                            founded_lengths.append(
                                (len_now - len_dijkstra_start) * 2 + len_dijkstra_start)  # len of backpath and return

                        # count repeats but only 1 edge - not complete
                        n = int((upper - len_dijkstra_start) / int(2 * v))  # n times wiederholungen

                        path_now_copy = copy.deepcopy(path_now)
                        path_now_copy += [k]
                        length_to_node = v

                        len_total = (len_now - len_dijkstra_start) * n + len_dijkstra_start
                        path_repeats = [path_now_copy[-2]] + [k]

                        path_repeats1 = [path_repeats[0]] + path_repeats[::-1]
                        path_new = path_d_unchange[:-1] + path_repeats * int(n / 2) + [path_now_copy[-2]]

                        founded_paths.append(path_new)
                        # path_repeats =  set(path_now_copy) - set(path_d_unchange[:-1])


                        #TODO count repeats but only edges mit n nodes between
                        n1 = int((upper - len_dijkstra_start) / int(
                            2 * (len_now - len_dijkstra_start)))  # n times wiederholungen

                        path_now_copy1 = copy.deepcopy(path_now)


                    else:
                        pass

                if v * 2 + len_now <= upper:  #TODO wiederholungen - schleifen
                    n = int((upper - len_dijkstra_start) / int(2 * v))  # n times wiederholungen

                # rekursiv weiter
                find_reversed(k, v, path_now, len_now, upper, visited_nodes, sink, founded_paths, new_path)
        else:
            if len_dijkstra_start + v <= upper:
                pass
                # TODO founded_paths.append(path_now + new_path[:-1] + new_path[::-1] + [sink])
                # list(set(founded_paths))
            visited_nodes.clear()

            return

    for el in graph[path[-1]].items():
        find_reversed(el[0], el[1], path, distances, upper, visited_nodes, sink, founded_paths, new_path=[])

    return (tuple(zip(founded_paths, founded_lengths)))

# works on assumption that another shortest paths are near from the shortest path
# alternative to Yen's Algorithm - Yen's A doesnt find n the most shortest paths, but any n shortest paths

# graph - input graph
# path - dijkstra shortest path
# source - start position in graph
# sink - end position in graph
# upper - max length of path
def find_in_dijkstra(graph, path, source, sink, upper): #1-2-4 #len = 20 # easy Yen's
    found_paths = []
    found_lengths = []
    for el in path[1:-1]: # for every node between source and sink
        graph_clone = copy.deepcopy(graph) # make copy of the graph
        del graph_clone[el] # del 1 Node between
        for element in graph_clone.values(): # delete edges to another nodes from this node
            element.pop(el, None)
        path, distance = dijkstra(graph_clone, source, sink)
        if distance <= upper: # greedy, looking for another paths but returns only that less then upper bound - len tolerance
            found_paths.append(path)
            found_lengths.append(distance)
    return (tuple(zip(found_paths,found_lengths)))
# it doesnt handels the case wenn 2 or more nodes deleted - in Yens, but Yens is unpredictable


# Usage example
n1, n2, n3, n4 = Node(1), Node(2), Node(3), Node(4)
## ein demo_graph immer auskommentieren, sonst proplemen mit adjacenten kanten


demo_graph = UndirectedGraph(
    [
        UndirectedEdge((n1, n2), 10),
        UndirectedEdge((n1, n3), 30),
        UndirectedEdge((n2, n4), 10),
        UndirectedEdge((n3, n4), 10),
    ]
)

n5, n6, n7 = Node(5), Node(6), Node(7)
# demo_graph = UndirectedGraph(
#     [
#         UndirectedEdge((n1, n2), 10),
#         UndirectedEdge((n1, n3), 30),
#         UndirectedEdge((n2, n4), 10),
#         UndirectedEdge((n3, n4), 10),
#         UndirectedEdge((n4, n5), 10),
#         UndirectedEdge((n4, n6), 30),
#         UndirectedEdge((n6, n7), 10),
#         UndirectedEdge((n5, n7), 10),
#     ]
# )

###############
# Should print the path [1, 2, 4]
#print(compute_shortest_paths(demo_graph, n1, n4, 1.0))

# Should print the paths [1, 2, 4], [1, 3, 4], [1, 2, 4, 2, 4], [1, 2, 1, 2, 4], [1, 2, 4, 3, 4]
#print(compute_shortest_paths(demo_graph, n1, n4, 2.0))
###############

# immer 1 print auskommentieren und pass keyword weg
if __name__ == '__main__':

    # Case 1: demo_graph mit 4 Knoten - default graph

    print(compute_shortest_paths(demo_graph, n1, n4, 4.0))
    # Output:
    #1-2-4
    #1-2-4-2-4
    #1-2-4-3-4
    #1-3-4

    #print(compute_shortest_paths(demo_graph, n1, n4, 1.0))
    # Output:
    #1-2-4

    #print(compute_shortest_paths(demo_graph, n1, n4, 8.0))
    # Output:
    # 1-2-4
    # 1-2-4-2-4
    # 1-2-4-2-4-2-4-2-4
    # 1-3-4



    #Case 2: demo_graph mit 7 Knoten

    #print(compute_shortest_paths(demo_graph, n1, n7, 4.0))
    # Output:
    #1-2-4-5-7 :len 40 shortest path - true
    #1-2-4-5-7-6-7 :len 60 - passt
    #1-2-4-5-7-5-7 : len 60 - passt
    #1-3-4-5-7 : len 60 - passt
    #1-2-4-6-7 : len 60 - passt

    #print(compute_shortest_paths(demo_graph, n1, n7, 1.0))
    # Output:
    #1-2-4-5-7