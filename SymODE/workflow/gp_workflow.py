import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()

nodes = [
    "Population init",
    "Variation",
    "Hard-screening",
    "Fitness evaluation",
    "Selection",
    "Termination?"
]

G.add_edges_from([
    ("Population init", "Variation"),
    ("Variation", "Hard-screening"),
    ("Hard-screening", "Fitness evaluation"),
    ("Fitness evaluation", "Selection"),
    ("Selection", "Termination?"),
    ("Termination?", "Variation"),   # feedback loop
])

pos = {
    "Population init": (0, 5),
    "Variation": (0, 4),
    "Hard-screening": (0, 3),
    "Fitness evaluation": (0, 2),
    "Selection": (0, 1),
    "Termination?": (0, 0)
}

plt.figure(figsize=(6, 8))
nx.draw(
    G, pos,
    with_labels=True,
    node_size=3000,
    node_color="white",
    edgecolors="black",
    font_size=10,
    arrowsize=20,
    arrowstyle="->"
)
plt.title("GP Loop Flowchart")
plt.axis("off")
plt.tight_layout()
plt.savefig("gp_loop_networkx.png", dpi=300)
plt.show()
