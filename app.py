
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from task3 import (
    G,
    coords,
    gbfs,
    astar,
    path_cost,
    START,
    GOAL
)


st.set_page_config(
    page_title="Hospital Emergency Supply Robot",
    layout="centered"
)


st.title("Hospital Emergency Supply Robot")
st.caption(
    "Interactive visualization of GBFS and A* Search"
)


# ============================================================
# User selections
# ============================================================

nodes = list(G.nodes)

col1, col2, col3 = st.columns(3)

start = col1.selectbox(
    "Initial Node",
    nodes,
    index=nodes.index(START)
)

goal = col2.selectbox(
    "Goal Node",
    nodes,
    index=nodes.index(GOAL)
)

algorithm = col3.selectbox(
    "Search Algorithm",
    ["GBFS", "A*"]
)


# ============================================================
# Run search
# ============================================================

if st.button("Run Search", type="primary"):

    if start == goal:

        st.warning(
            "Initial node and goal node must be different."
        )

    else:

        if algorithm == "GBFS":
            path, expansion = gbfs(G, start, goal)

        else:
            path, expansion = astar(G, start, goal)

        st.session_state["result"] = {
            "start": start,
            "goal": goal,
            "algorithm": algorithm,
            "path": path,
            "expansion": expansion
        }


# ============================================================
# Display graph
# ============================================================

result = st.session_state.get("result")

pos = nx.get_node_attributes(G, "pos")

path = None

if result:
    path = result["path"]

path_edges = []

if path:
    path_edges = list(zip(path, path[1:]))


def get_node_color(node):

    if result and node == result["start"]:
        return "lightcoral"

    if result and node == result["goal"]:
        return "gold"

    if path and node in path:
        return "lightgreen"

    return "skyblue"


node_colors = [
    get_node_color(node)
    for node in G.nodes
]


fig, ax = plt.subplots(figsize=(10, 7))

nx.draw(
    G,
    pos,
    ax=ax,
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
    ax=ax,
    edgelist=path_edges,
    edge_color="red",
    width=3,
    arrowsize=25
)

nx.draw_networkx_edge_labels(
    G,
    pos,
    ax=ax,
    edge_labels=nx.get_edge_attributes(G, "weight")
)

st.pyplot(fig)


# ============================================================
# Display results
# ============================================================

if result:

    st.subheader("Search Result")

    st.write(
        f"**Selected Algorithm:** {result['algorithm']}"
    )

    if result["path"]:

        st.write(
            "**Solution Path:** "
            + " → ".join(result["path"])
        )

        total = path_cost(
            G,
            result["path"]
        )

        st.write(
            f"**Total Path Cost:** {total:.2f}"
        )

        st.write(
            "**Node Expansion Sequence:** "
            + " → ".join(result["expansion"])
        )

        # ----------------------------------------------------
        # A* g(n), h(n), f(n) table
        # ----------------------------------------------------

        if result["algorithm"] == "A*":

            st.subheader(
                "A* Node Evaluation Table"
            )

            st.table(result["expansion"])

    else:

        st.error(
            f"No path exists from {result['start']} "
            f"to {result['goal']}."
        )

else:

    st.info(
        "Select an initial node, goal node, and search "
        "algorithm, then click Run Search."
    )


# ============================================================
# Explanation
# ============================================================

st.subheader("GBFS vs A*")

st.write(
    "**GBFS:** Uses only h(n), the estimated distance "
    "from the current node to the goal. It therefore "
    "focuses on the node that appears closest to the goal."
)

st.write(
    "**A*:** Uses f(n) = g(n) + h(n). It considers both "
    "the actual cost already travelled and the estimated "
    "remaining cost."
)

