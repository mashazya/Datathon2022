import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import distance, distance_matrix
from sklearn.cluster import KMeans
from io import StringIO
from custom_parser import parse_file
from joblib import Parallel, delayed
# ------------------------------------------------------------------------------
#                             Functions definition
# ------------------------------------------------------------------------------

def clustering_algorithm(df, k=16):
    pins_df = df[df.driver_type == 0].reset_index(drop=True)
    X = pins_df[['x', 'y']].to_numpy()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    pins_df['labels'] = kmeans.labels_

    tsp = nx.approximation.traveling_salesman_problem
    clusters = []

    #results = Parallel(n_jobs=2)(delayed(countdown)(10**7) for _ in range(20))

    def get_cluster(i):
        indexes = pins_df[pins_df.labels == i].index.values
        if len(indexes) == 1:
            cluster = [i] + [indexes[0]] + [i]
        else:
            dist_matrix = distance_matrix(pins_df[pins_df.labels == i].loc[:,['x', 'y']].to_numpy(), pins_df[pins_df.labels == i].loc[:,['x', 'y']].to_numpy(), p=1)
            G = nx.from_numpy_matrix(dist_matrix)
            chain = tsp(G, cycle=False)
            chain = [indexes[x] + 1 for x in chain]
            cluster = [i] + chain + [i]
        return cluster

    clusters = []
    for i in np.sort(pins_df.labels.unique()):
        clusters.append(get_cluster(i))
    return clusters

def create_solution_edges(example_sol, example_sol_idx):
        global G
        np.random.seed(1)
        xs = [v for v in nx.get_node_attributes(G, "x").values()]
        ys = [v for v in nx.get_node_attributes(G, "y").values()]
        for i, chain in enumerate(example_sol):
            for j in range(len(chain)-1):
                u,v = chain[j], chain[j+1]
                if np.random.rand() > 0.5:  # add dummy edges
                    dummy_x = xs[example_sol_idx[i][j]]
                    dummy_y = ys[example_sol_idx[i][j+1]]
                else:
                    dummy_x = xs[example_sol_idx[i][j+1]]
                    dummy_y = ys[example_sol_idx[i][j]]
                dummy_name = 'dummy_' + str(u) + '-' + str(v)
                G.add_node(dummy_name, pos=(dummy_x, dummy_y), driver_type=-1)
                G.add_edge(u, dummy_name)
                G.add_edge(dummy_name,v)
                np.append(example_sol[i], dummy_name)

def get_k_random_rgb_color(k):
    np.random.seed(1)
    colors = []
    levels = range(32,256,32)
    for i in range(k):
        colors.append([np.random.choice(levels)/256 for _ in range(3)] + [1])
    return colors

available_colors = [
    (25,25,112),
    (72,61,139),
    (0,0,255),
    (123,104,238),
    (135,206,250),
    (70,130,180),
    (147,112,219),
    (75,0,130),
    (186,85,211),
    (221,160,221),
    (216,191,216),
    (255,0,255),
    (255,182,193),
    (219,112,147),
    (0,0,128),
    (255,105,80)
]
for i, color in enumerate(available_colors):
    available_colors[i] = tuple(v/256 for v in color)

available_colors = get_k_random_rgb_color(16)

def get_partition(node, edge_partitions):
    for i, chain in enumerate(edge_partitions):
        if node in chain:
            return i
    return -1

def get_position(G, position_attribute='pos'):
    return {pin:(x, y) for (pin, (x,y)) in nx.get_node_attributes(G, position_attribute).items()}

def get_node_sizes(G):
    sizes = []
    nodes_driver_type = nx.get_node_attributes(G, 'driver_type')
    for node in (G.nodes()):
        if nodes_driver_type[node] == -1:
            sizes.append(0)
        else:
            sizes.append(15)
    return sizes

def get_node_color(G, edge_partitions=[[None]], driver_type_column='driver_type', chains=16):
    global available_colors
    color_map = []
    for n, driver_type in nx.get_node_attributes(G, driver_type_column).items():
        if driver_type == 0:  # pin
            partition = get_partition(n, edge_partitions)
            if chains==len(edge_partitions) or chains==partition:
                color_map.append(available_colors[partition])
            else:
                color_map.append((0,0,0,0.1))
        elif driver_type == 1:  #input
            color_map.append('green') 
        elif driver_type == 2:
            color_map.append('red') 
        else:
            color_map.append('white')
    return color_map

def get_edge_color(G, edge_partitions=[[None]], chains=16):
    global available_colors
    edge_colors = []
    for u,_ in G.edges():
        i = get_partition(u, edge_partitions)
        if i == -1 or (chains != len(edge_partitions) and chains != i):
            edge_colors.append((0,0,0,0.1))
        else:
            edge_colors.append(available_colors[i])    
    return edge_colors

def get_bar_colors(chains=16, all_edges_chain=16):
    global available_colors
    bar_colors = []
    for i, color in enumerate(available_colors):
        if chains != all_edges_chain and chains != i:
            bar_colors.append((0,0,0,0.1))
        else:
            bar_colors.append(color)    
    return bar_colors

def parse_output(data, net_name, routes):
    net_name = '- '+net_name
    with open('output.def', 'w') as f:
        for route in routes:
            f.writelines([net_name,'\n'])
            f.writelines(['  ( ', data['name_pin'].tolist()[route[0]], ' conn_in )\n'])
            f.writelines(['  ( ', data['name_pin'].tolist()[route[1]], ' conn_out )\n'])
            f.write(';\n')
            for i in range (1,len(route)-2):
                f.writelines([net_name,'\n'])
                f.writelines(['  ( ', data['name_pin'].tolist()[route[i]], ' conn_in )\n'])
                f.writelines(['  ( ', data['name_pin'].tolist()[route[i+1]], ' conn_out )\n'])
                f.write(';\n')

            f.writelines([net_name,'\n'])
            f.writelines(['  ( ', data['name_pin'].tolist()[route[-2]], ' conn_in )\n'])
            f.writelines(['  ( ', data['name_pin'].tolist()[route[-1]], ' conn_out )\n'])
            f.write(';\n')
        f.close()

def calculate_metric(routes, df_pins, drivers):
    routes_length = []
    for route in routes:
        route_distances = []

        for i in range(0,len(route)):
            if i == 0:
                driver_in = drivers.iloc[route[i]].values[1:3]
                pin = df_pins.loc[route[i+1]-31].values[1:3]  # pins in df_pins go from 1 to n
                route_distances.append(distance.minkowski(driver_in, pin, 1))
            elif i == len(route)-2:
                driver_out = drivers.iloc[route[i+1]].values[1:3]
                pin = df_pins.loc[route[i]-31].values[1:3]  # pins in df_pins go from 1 to n
                route_distances.append(distance.minkowski(driver_out, pin, 1))
                break
            else:
                route_distances.append(distance.minkowski(df_pins.loc[route[i]-31].values[1:3], df_pins.loc[route[i+1]-31].values[1:3])) # pins in df_pins go from 1 to n
                 
        routes_length.append(sum(route_distances))
    return sum(routes_length) / len(routes_length), np.std(routes_length), routes_length


# ------------------------------------------------------------------------------
#                                    APP
# ------------------------------------------------------------------------------
example_sol_idx = None  # Streamlit ptimization purposes

# File upload
uploaded_file = st.file_uploader(
    'Upload your input .def file:',
    accept_multiple_files=False,
    type=['def', 'txt'],

)
if uploaded_file is not None:
    str_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = parse_file(str_file)
  
    # Generate graph from parsed input file
    pins = [
        (
            row.name_pin,
            {
                "pos": (row.x, row.y),
                "x": row.x,
                "y": row.y,
                "driver_type": row.driver_type
            }
        )
        for _, row in df.iterrows()
    ]
    G = nx.Graph()
    G.add_nodes_from(pins)

    if example_sol_idx is None:
        # Define solution
        with st.spinner("Computing solution"):
            example_sol_idx = clustering_algorithm(df)
            st.balloons()  # :)
    
    for i in range(len(example_sol_idx)):
        for j in range(len(example_sol_idx[i])):
            if 0 < j < len(example_sol_idx[i])-1:
                example_sol_idx[i][j] += 31
            elif j == len(example_sol_idx[i])-1:
                example_sol_idx[i][j] += 16

    example_sol = [df.loc[items, 'name_pin'].values for items in example_sol_idx]

    # Add edges to graph
    create_solution_edges(example_sol, example_sol_idx)

    # Selectbox
    options = np.array([f'Chain {i}' for i in range(len(example_sol))] + ['All chains'])
    selected_option = st.selectbox('Choose a chain to highlight:', options)
    selected_chains = np.where(options == selected_option)[0][0]

    f_pins = plt.figure()
    nx.draw(
        G,
        get_position(G),
        node_size=get_node_sizes(G), 
        node_color=get_node_color(G, example_sol, chains=selected_chains),
        edge_color=get_edge_color(G, example_sol, chains=selected_chains),
    )
    st.pyplot(fig=f_pins)

    # Compute solution metrics
    driver_centr_x = int((df[df.driver_type != 0].x).mean())
    driver_centr_y = int((df[df.driver_type != 0].y).mean())
    drivers = df[df['driver_type'] != 0]
    df_pins = df.drop([i for i in range(32)])
    df_pins.loc[-1] = ['depot', driver_centr_x, driver_centr_y, 1]
    df_pins.index = df_pins.index + 1  # shifting index
    df_pins = df_pins.sort_index().reset_index(drop=True)
    avg_dist, std, routes_distances = calculate_metric(example_sol_idx, df_pins, drivers)

    # Plot distances
    f_distances = plt.figure()
    plt.barh(
        [i for i in range(len(routes_distances))], 
        routes_distances, 
        color=get_bar_colors(selected_chains, all_edges_chain=len(example_sol)), 
        label=[i for i in range(len(routes_distances))]
    )
    plt.yticks(np.arange(0, len(routes_distances), 1.0))
    plt.xticks(np.arange(0, max(routes_distances), max(routes_distances)/10))
    plt.axvline(x=avg_dist, ymin=0, ymax=1, color='black')
    plt.title('Distances per chain')
    plt.xlabel('Total distance')
    plt.ylabel('Chains')
    plt.text(x=avg_dist+max(routes_distances)*0.01, y=len(routes_distances)-1, s='Average distance')
    st.pyplot(f_distances)

    output_name = st.text_input('Name of your chip outut file:')
    parse_output(df, output_name, example_sol_idx)
    col1,col2,col3 = st.columns(3)
    with col2:
        with open('./output.def', "rb") as file:
            st.download_button(
                label='Download output file',
                data = file,
                file_name = output_name + '_output.def'
            )

else:  # no file added
    st.write('Please add an input .def file 🙂')