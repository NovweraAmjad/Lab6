import math
import heapq
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Weighted hospital graph and coordinates
# ============================================================

coords = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6),
}

edges = [
    ("Pharmacy", "Patient_Wing", 4.1),
    ("Pharmacy", "Main_Corridor", 2.2),
    ("Main_Corridor", "Nursing_Station", 2.2),
    ("Patient_Wing", "Laboratory", 5.0),
    ("Nursing_Station", "Laboratory", 3.2),
    ("Nursing_Station", "Emergency_Ward", 6.0),
    ("Laboratory", "Emergency_Ward", 3.2),
]

G = nx.DiGraph()

for node, position in coords.items():
    G.add_node(node, pos=position)

for u, v, weight in edges:
    G.add_edge(u, v, weight=weight)


START = "Pharmacy"
GOAL = "Emergency_Ward"


# ============================================================
# 2. Euclidean heuristic
# ============================================================

def heuristic(node, goal):
    x1, y1 = coords[node]
    x2, y2 = coords[goal]

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ============================================================
# 3. Path reconstruction
# ============================================================

def reconstruct_path(parent, node):
    path = []

    while node is not None:
        path.append(node)
        node = parent[node]

    return path[::-1]


# ============================================================
# 4. Calculate path cost
# ============================================================

def path_cost(graph, path):
    total = 0

    for u, v in zip(path, path[1:]):
        total += graph[u][v]["weight"]

    return total


# ============================================================
# 5. Greedy Best-First Search
# ============================================================

def gbfs(graph, start, goal):
    frontier = [(heuristic(start, goal), start)]

    parent = {start: None}
    visited = set()
    expansion_order = []

    while frontier:
        _, node = heapq.heappop(frontier)

        if node in visited:
            continue

        visited.add(node)
        expansion_order.append(node)

        if node == goal:
            path = reconstruct_path(parent, node)
            return path, expansion_order

        for neighbor in graph.neighbors(node):
            if neighbor not in visited and neighbor not in parent:
                parent[neighbor] = node
                heapq.heappush(
                    frontier,
                    (heuristic(neighbor, goal), neighbor)
                )

    return None, expansion_order


# ============================================================
# 6. A* Search
# ============================================================

def astar(graph, start, goal):
    frontier = [
        (heuristic(start, goal), 0.0, start)
    ]

    g_cost = {start: 0.0}
    parent = {start: None}

    closed = set()
    expansion = []

    while frontier:
        f, g, node = heapq.heappop(frontier)

        if node in closed or g > g_cost[node]:
            continue

        closed.add(node)

        h = heuristic(node, goal)

        expansion.append({
            "Node": node,
            "g(n)": round(g, 2),
            "h(n)": round(h, 2),
            "f(n)": round(f, 2)
        })

        if node == goal:
            path = reconstruct_path(parent, node)
            return path, expansion

        for neighbor in graph.neighbors(node):
            new_g = g + graph[node][neighbor]["weight"]

            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                parent[neighbor] = node

                closed.discard(neighbor)

                new_f = new_g + heuristic(neighbor, goal)

                heapq.heappush(
                    frontier,
                    (new_f, new_g, neighbor)
                )

    return None, expansion


# ============================================================
# 7. Run A* for the assignment
# ============================================================

if __name__ == "__main__":

    path, expansion = astar(G, START, GOAL)

    print("Node Expansion Sequence:")

    for i, row in enumerate(expansion, 1):
        print(f"{i}. {row['Node']}")

    if path:
        total = path_cost(G, path)

        print("\nSolution Path:")
        print(" -> ".join(path))

        print("\nTotal Path Cost:")
        print(round(total, 2))

        print("\nA* Table:")

        table = pd.DataFrame(expansion)
        table.index = range(1, len(table) + 1)
        table.index.name = "Order"

        print(table)

        # NetworkX visualization
        pos = nx.get_node_attributes(G, "pos")

        path_edges = list(zip(path, path[1:]))

        node_colors = []

        for node in G.nodes:
            if node == START:
                node_colors.append("lightcoral")
            elif node == GOAL:
                node_colors.append("gold")
            elif node in path:
                node_colors.append("lightgreen")
            else:
                node_colors.append("skyblue")

        plt.figure(figsize=(10, 7))

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=2500,
            font_size=8,
            arrowsize=20,
            edgecolors="black"
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            edge_color="red",
            width=3,
            arrowsize=25
        )

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=nx.get_edge_attributes(G, "weight")
        )

        plt.title(
            f"A* Path: {' -> '.join(path)} | Cost = {total:.2f}"
        )

        plt.show()

