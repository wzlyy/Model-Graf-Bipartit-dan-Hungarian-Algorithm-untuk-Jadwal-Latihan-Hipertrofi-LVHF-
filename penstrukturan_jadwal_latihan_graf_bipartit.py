import networkx as nx
import matplotlib.pyplot as plt

# 1. Himpunan U (muscle targets)
muscles = [
    'Anterior Delt', 'Medial Delt', 'Posterior Delt', 'Upper Chest',
    'Middle Chest','Lower Chest', 'Latissimus Dorsi', 'Rhomboid', 
    'Quadriceps', 'Hamstring', 'Tricep', 'Bicep'
]
# 2. Himpunan V (exercise options)
exercises = [
    'Lateral Raise', 'Frontal Raise', 'Shoulder Press', 'Reverse Fly',
    'Flat Bench Press', 'Chest Fly','Incline Bench Press', 'Lat Pulldown', 
    'Seated Row', 'Keenan Flaps','Squat','Leg Extensions',
    'Leg Curl', 'Nordic Curl', 'Cable Tricep Extension', 'Preacher Curl'
]
# 3. Mechanical Tension score berdasarkan %MVIC
mt_scores = [
    [0.4,0.9,0.7,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Anterior Delt
    [0.7,0.3,0.6,0.4,0,0,0,0,0,0,0,0,0,0,0,0],  # Medial Delt
    [0.5,0.2,0.2,0.65,0,0,0,0,0,0,0,0,0,0,0,0],  # Posterior Delt
    [0,0,0,0,0.27,0.25,0.3,0,0,0,0,0,0,0,0,0],  # Upper Chest
    [0,0,0,0,0.28,0.41,0.19,0,0,0,0,0,0,0,0,0],  # Middle Chest
    [0,0,0,0,0.25,0.40,0.15,0,0,0,0,0,0,0,0,0],  # Lower Chest
    [0,0,0,0,0,0,0,0.26,0.37,0.48,0,0,0,0,0,0],  # Latissimus Dorsi
    [0,0,0,0,0,0,0,0.22,0.35,0.1,0,0,0,0,0,0],  # Rhomboid
    [0,0,0,0,0,0,0,0,0,0,0.55,0.6,0,0,0,0],     # Quadriceps
    [0,0,0,0,0,0,0,0,0,0,0.22,0.35,0.64,0.9,0,0], # Hamstring
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.7,0], # Tricep
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.75] # Bicep
]
# 4. Buat edge berbobot
edges = []
for i in range(len(muscles)):
    for j in range(len(exercises)):
        score = mt_scores[i][j]
        if score > 0:
            edges.append((muscles[i], exercises[j], score))
# 5. Buat graf
G = nx.Graph()
G.add_nodes_from(muscles, bipartite=0)
G.add_nodes_from(exercises, bipartite=1)
G.add_weighted_edges_from(edges)


# Posisi untuk visualisasi bipartit dengan jarak vertikal yang lebih lebar
vertical_spacing = 2.0
horizontal_spacing = 0.2
pos = {}
pos.update((node, (0, i *2* vertical_spacing)) for i, node in enumerate(muscles))
pos.update((node, (horizontal_spacing, i *vertical_spacing)) for i, node in enumerate(exercises))

# Create a very tall figure that can be scrolled
fig = plt.figure(figsize=(12, 40))  # Width, Height (much taller)

# Draw the graph
nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=2500)
nx.draw_networkx_labels(G, pos, font_size=10)
nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5)

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)


import numpy as np
from scipy.optimize import linear_sum_assignment
# Buat matriks biaya (negasi dari skor karena Hungarian mencari minimisasi)
cost_matrix = np.ones((len(muscles), len(exercises)))  

for i in range(len(muscles)):
    for j in range(len(exercises)):
        score = mt_scores[i][j]
        if score > 0:
            cost_matrix[i][j] = 1 - score  #Semakin tinggi skor, semakin rendah biaya

# Jalankan algoritma Hungarian (Linear Sum Assignment)
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Ambil hasil assignment optimal
assignments = [(muscles[i], exercises[j], mt_scores[i][j])
                for i, j in zip(row_ind, col_ind) if mt_scores[i][j] > 0]
# Buat graf bipartit hasil assignment saja
G_assign = nx.Graph()
G_assign.add_nodes_from(muscles, bipartite=0)
G_assign.add_nodes_from(exercises, bipartite=1)
G_assign.add_weighted_edges_from([(m, e, w) for m, e, w in assignments])

# Gunakan posisi yang sama seperti sebelumnya# Gambar graf hasil assignment
plt.figure(figsize=(28, 20))
nx.draw(
    G_assign, pos, with_labels=True, node_color='lightcoral',
    node_size=3000, font_size=10, edge_color='black', width=2
)
edge_labels = {(m, e): f"{w:.2f}" for m, e, w in assignments}
nx.draw_networkx_edge_labels(G_assign, pos, edge_labels=edge_labels, font_size=8)
plt.axis('off')
plt.tight_layout()
plt.show()

