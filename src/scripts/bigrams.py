import math
import networkx as nx
from bokeh.palettes import Spectral4
from bokeh.models import (
    BoxSelectTool,
    Circle,
    HoverTool,
    MultiLine,
    NodesAndLinkedEdges,
    TapTool,
    ColumnDataSource,
    LabelSet,
)
from bokeh.plotting import from_networkx
from bokeh.transform import linear_cmap
from typing import List, Dict


def scale(data: Dict[int]) -> Dict[int]:
    """
    Calculates logarithm base 2 and multiplies the result by 3.
    Args:
        data (Dict[int]): A ditctionary with values.
    Returns:
        Dict[int]: The input dictonary with updated values.
    """

    for key, value in data.items():
        data[key] = (math.log2(value)) * 3

    return data


def bigram_freq(freq: Dict[str, int], G, scale: bool) -> Dict[str, int]:
    """
    Creates a dictionary of words (present in bigrams) and their frequencies.
    Args:
        freq (Dict[str, int]): A dictionary with all words from the data corpus and their frequency values.
        G: A Networkx graph.
        scale (bool): If true, the logarithm of frequency values will be calculated.
    Returns:
        Dict[str, int]: The dictionary containg words from bigrams and their frequencies.
    """

    freq_dict = {}

    for node in G.nodes():
        for word in freq:
            if word[0] == node:
                if scale == True:
                    freq_dict[word[0]] = (math.log2(word[1])) * 3
                else:
                    freq_dict[word[0]] = word[1]
    return freq_dict


def plot_bigrams(
    G,
    word_freq: Dict[str, int],
    co_occurence: Dict[str, int],
    pos: Dict[int],
    palette_nodes: List[str],
    palette_edges: List[str],
    title: str,
):
    """
    Plots a node graph.
    Args:
        G: A Networkx graph.
        word_freq (Dict[str, int]):  A dictionary of bigram members and their frequencies across the dataset.
        co_occurence (Dict[str, int]): A dictionaty of bigrams and their co-occurence values.
        pos (Dict[int]): A dictionary of positions keyed by a node.
        palette_nodes (List[str]): A colour palette for the nodes.
        palette_edges (List[str]):  A colour palette for the edges.
        title (str): The title of the node graph.
    Returns:
        plot: The node graph.
    """

    from bokeh.plotting import figure

    nx.set_node_attributes(G, name="freq", values=word_freq)
    nx.set_edge_attributes(G, name="co_occurence", values=co_occurence)

    node_highlight_color = Spectral4[1]
    edge_highlight_color = Spectral4[2]

    color_nodes = "freq"
    color_edges = "co_occurence"

    plot = figure(
        tools="pan,wheel_zoom,save,reset", active_scroll="wheel_zoom", title=title
    )

    plot.title.text_font_size = "20px"

    plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())

    network_graph = from_networkx(G, pos, scale=10, center=(0, 0))

    min_col_val_node = min(network_graph.node_renderer.data_source.data[color_nodes])
    max_col_val_node = max(network_graph.node_renderer.data_source.data[color_nodes])

    min_col_val_edge = min(network_graph.edge_renderer.data_source.data[color_edges])
    max_col_val_edge = max(network_graph.edge_renderer.data_source.data[color_edges])

    network_graph.node_renderer.glyph = Circle(
        size=40,
        fill_color=linear_cmap(
            color_nodes, palette_nodes, min_col_val_node, max_col_val_node
        ),
        fill_alpha=1,
    )

    network_graph.node_renderer.hover_glyph = Circle(
        size=5, fill_color=node_highlight_color, line_width=3
    )

    network_graph.node_renderer.selection_glyph = Circle(
        size=5, fill_color=node_highlight_color, line_width=5
    )

    network_graph.edge_renderer.glyph = MultiLine(
        line_alpha=1,
        line_color=linear_cmap(
            color_edges, palette_edges, min_col_val_edge, max_col_val_edge
        ),
        line_width=4,
    )

    network_graph.edge_renderer.selection_glyph = MultiLine(
        line_color=edge_highlight_color, line_width=4
    )
    network_graph.edge_renderer.hover_glyph = MultiLine(
        line_color=edge_highlight_color, line_width=4
    )

    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(network_graph)

    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource(
        {"x": x, "y": y, "name": [node_labels[i] for i in range(len(x))]}
    )
    labels = LabelSet(
        x="x",
        y="y",
        text="name",
        source=source,
        background_fill_color="pink",
        text_font_size="24px",
        background_fill_alpha=0.3,
    )

    plot.renderers.append(labels)

    return plot
