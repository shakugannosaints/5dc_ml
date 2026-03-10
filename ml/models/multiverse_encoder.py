# ml/models/multiverse_encoder.py
"""
Multiverse GNN Encoder: Graph Attention Network over the dynamic multiverse graph.

The multiverse in 5D chess is a DAG of boards connected by:
  - Temporal edges (same timeline, consecutive turns)
  - Branch edges (timeline splits from move to another timeline)

Each node = one board position (l, t, color).
Node features = board embedding (from BoardEncoder) + scalar features.
Edge features = edge type (temporal=0, branch=1).

Output:
  - Per-node embeddings (for policy)
  - Global graph embedding (for value)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch

from ..config import MultiverseEncoderConfig


class TypedGATLayer(nn.Module):
    """
    A single GAT layer that handles typed edges.
    We use edge_attr as a learned embedding added to attention.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, num_edge_types: int):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=num_edge_types,     # edge features dimension
            add_self_loops=True,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [N, in_dim]
            edge_index: [2, E]
            edge_attr:  [E, num_edge_types] one-hot edge type
        Returns:
            [N, out_dim]
        """
        out = self.gat(x, edge_index, edge_attr=edge_attr)
        out = self.norm(out)
        return F.elu(out)


class MultiverseEncoder(nn.Module):
    """
    Encodes the full multiverse graph into:
      - node_embeds: [N, gnn_hidden_dim]  per-board embeddings
      - global_embed: [global_dim]         single vector for the full state
    
    Pipeline:
      1. Project board embeddings + scalar features → input node features
      2. Multi-layer GAT with typed edges
      3. Global pooling → linear → global embedding
    """

    def __init__(self, cfg: MultiverseEncoderConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = MultiverseEncoderConfig()
        self.cfg = cfg

        # Node input projection: board_embed + scalars → gnn_hidden
        node_input_dim = cfg.board_embed_dim + cfg.node_scalar_dim
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, cfg.gnn_hidden_dim),
            nn.LayerNorm(cfg.gnn_hidden_dim),
            nn.ELU(),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(cfg.gnn_num_layers):
            self.gat_layers.append(
                TypedGATLayer(
                    in_dim=cfg.gnn_hidden_dim,
                    out_dim=cfg.gnn_hidden_dim,
                    num_heads=cfg.gnn_num_heads,
                    num_edge_types=cfg.num_edge_types,
                )
            )

        # Residual projection (identity if dims match)
        # Already matching: gnn_hidden_dim -> gnn_hidden_dim

        # Global pooling → global embedding
        self.global_proj = nn.Sequential(
            nn.Linear(cfg.gnn_hidden_dim, cfg.global_dim),
            nn.LayerNorm(cfg.global_dim),
            nn.ELU(),
        )

    def forward(self, board_embeds: torch.Tensor, node_scalars: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor,
                batch: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            board_embeds: [N, board_embed_dim]   from BoardEncoder
            node_scalars: [N, node_scalar_dim]   scalar features per node
            edge_index:   [2, E]                 graph connectivity
            edge_type:    [E]                    int type per edge (0=temporal, 1=branch)
            batch:        [N]                    batch assignment (for batched graphs), or None for single graph
        
        Returns:
            node_embeds:  [N, gnn_hidden_dim]    per-node embeddings
            global_embed: [B, global_dim]        per-graph global embedding (B=1 if single graph)
        """
        # One-hot encode edge types → [E, num_edge_types]
        edge_attr = F.one_hot(edge_type.long(), self.cfg.num_edge_types).float()

        # Concatenate board embeddings and scalar features
        x = torch.cat([board_embeds, node_scalars], dim=-1)  # [N, board_embed + scalar]
        x = self.node_proj(x)  # [N, gnn_hidden]

        # GAT layers with residual connections
        for gat_layer in self.gat_layers:
            x_new = gat_layer(x, edge_index, edge_attr)
            x = x + x_new  # residual

        node_embeds = x  # [N, gnn_hidden_dim]

        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        global_embed = global_mean_pool(x, batch)  # [B, gnn_hidden_dim]
        global_embed = self.global_proj(global_embed)  # [B, global_dim]

        return node_embeds, global_embed

    def build_graph_from_engine(self, graph_struct: dict, board_embeds_dict: dict[tuple, torch.Tensor],
                                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Convert engine's graph_structure dict + board embeddings into GNN inputs.
        
        Args:
            graph_struct: from state.get_graph_structure()
                keys: node_keys, node_features, edge_src, edge_dst, edge_types
            board_embeds_dict: {(l,t,c): [embed_dim] tensor}
            device: target device
            
        Returns:
            board_embeds: [N, embed_dim]
            node_scalars: [N, scalar_dim]
            edge_index:   [2, E]
            edge_type:    [E]
            node_keys:    list of (l, t, c) tuples in order
        """
        node_keys = graph_struct['node_keys']   # list of (l, t, c) tuples
        node_features = graph_struct['node_features']  # list of lists (scalar features)
        edge_src = graph_struct['edge_src']
        edge_dst = graph_struct['edge_dst']
        edge_types = graph_struct['edge_types']

        n = len(node_keys)

        # Assemble board embeddings in node order
        embed_dim = next(iter(board_embeds_dict.values())).shape[-1] if board_embeds_dict else self.cfg.board_embed_dim
        board_embeds = torch.zeros(n, embed_dim, device=device)
        for i, key in enumerate(node_keys):
            k = tuple(key) if not isinstance(key, tuple) else key
            if k in board_embeds_dict:
                board_embeds[i] = board_embeds_dict[k]
            # else: zero embedding (for boards we don't have, shouldn't happen)

        # Node scalar features
        node_scalars = torch.tensor(node_features, dtype=torch.float32, device=device)  # [N, scalar_dim]
        # Pad or truncate to expected scalar_dim
        if node_scalars.shape[-1] < self.cfg.node_scalar_dim:
            pad = torch.zeros(n, self.cfg.node_scalar_dim - node_scalars.shape[-1], device=device)
            node_scalars = torch.cat([node_scalars, pad], dim=-1)
        elif node_scalars.shape[-1] > self.cfg.node_scalar_dim:
            node_scalars = node_scalars[:, :self.cfg.node_scalar_dim]

        # Edge index
        if len(edge_src) > 0:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
            edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_type = torch.zeros(0, dtype=torch.long, device=device)

        return board_embeds, node_scalars, edge_index, edge_type, node_keys
