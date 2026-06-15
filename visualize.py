import graphviz
from my_ml_lib.nn.autograd import Value  # Assuming Value is here
import numpy as np


# --- Implemented Graph Traversal ---
def get_all_nodes_and_edges(root_node: Value):
    """
    Performs a backward traversal from the root_node
    to find all unique Value nodes and the directed edges connecting them
    in the computation graph.

    Args:
        root_node (Value): The final node in the graph (e.g., the loss Value object).

    Returns:
        tuple: (nodes, edges)
               nodes (set): A set containing all Value objects found during traversal.
               edges (set): A set of tuples (parent_Value, child_Value) representing
                            the directed edges: parent -> child.
    """
    # --- Step 1 - Initialize sets ---
    nodes = set()
    edges = set()
    visited = set()

    # --- Step 2 - Recursive DFS Traversal ---
    def build_graph(v):
        if v in visited:
            return
        visited.add(v)

        # Add node to set
        nodes.add(v)

        # Traverse its parents and record edges
        for parent in getattr(v, "_prev", []):
            edges.add((parent, v))
            build_graph(parent)

    # --- Step 3 - Start Traversal ---
    build_graph(root_node)

    # --- Step 4 - Return Results ---
    return nodes, edges
# --- End Implementation ---


# --- Graph Drawing Function ---
def draw_dot(root_node: Value, format='svg', rankdir='LR'):
    """
    Generates a visualization of the computation graph using graphviz.
    Requires the `get_all_nodes_and_edges` function to be implemented correctly.

    Args:
        root_node (Value): The final node of the graph to visualize (e.g., loss).
        format (str): Output format ('svg', 'png', etc.). Default 'svg'.
        rankdir (str): Graph layout direction ('LR' or 'TB'). Default 'LR'.

    Returns:
        graphviz.Digraph: The graph object ready for rendering.
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = get_all_nodes_and_edges(root_node)

    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': rankdir})

    # Create nodes for Value objects
    for n in nodes:
        uid = str(id(n))  # Unique ID for the Value node

        # Format data and gradient strings
        data_str = (
            f"shape={n.data.shape}"
            if hasattr(n, "data") and isinstance(n.data, np.ndarray) and n.data.ndim > 0
            else f"{getattr(n, 'data', '?'):.4f}"
        )
        grad_str = (
            f"shape={n.grad.shape}"
            if hasattr(n, "grad") and isinstance(n.grad, np.ndarray) and n.grad.ndim > 0
            else f"{getattr(n, 'grad', '?'):.4f}"
        )
        label_str = f" | {getattr(n, 'label', '')}" if getattr(n, "label", "") else ""

        # Node label
        node_label = f"{{ data {data_str} | grad {grad_str}{label_str} }}"
        dot.node(name=uid, label=node_label, shape="record")

        # Add operation node if applicable
        op = getattr(n, "_op", "")
        if op:
            op_uid = uid + op
            dot.node(name=op_uid, label=op)
            dot.edge(op_uid, uid)

    # Add edges between Value nodes
    for n1, n2 in edges:
        parent_uid = str(id(n1))
        child_op = getattr(n2, "_op", "")
        if child_op:
            child_op_uid = str(id(n2)) + child_op
            dot.edge(parent_uid, child_op_uid)

    return dot


# --- Example Usage ---
if __name__ == '__main__':
    print("\n--- Visualization Example: Tiny MLP Forward Pass ---")
    from my_ml_lib.nn.modules.linear import Linear
    from my_ml_lib.nn.modules.activations import ReLU
    from my_ml_lib.nn.modules import Sequential
    from my_ml_lib.nn.autograd import Value

    # Define a small model: Linear(2,3) → ReLU
    model = Sequential(
        Linear(2, 3),
        ReLU()
    )

    # Create one sample input
    x = Value(np.array([[1.5, -2.0]]), label='x')  # shape (1, 2)

    # Forward pass
    output = model(x)
    output.label = "output"

    print("Generating example computation graph...")
    # Generate the graph visualization starting from the final node 'd'
    dot_graph = draw_dot(output)

    if dot_graph:
        # Render the graph to a file (e.g., 'example_graph.svg')
        # Requires Graphviz executables in system PATH
        try:
            output_filename = 'example_computation_graph'
            dot_graph.render(output_filename, view=False)
            print(f"Example graph saved as {output_filename}.* (e.g., .svg or .png)")
            print("Please include this generated graph in your report for Problem 4.")
        except graphviz.backend.execute.ExecutableNotFound:
            print("\n--- Graphviz Error ---")
            print("Graphviz executable not found. Visualization not saved.")
            print("Please install Graphviz (from www.graphviz.org)")
            print("and ensure the 'dot' command is available in your system's PATH.")
            print("----------------------\n")
        except Exception as e:
            print(f"An error occurred during graph rendering: {e}")
    else:
        print("Graph generation failed (likely due to traversal error).")
