import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Any, List, Dict, Tuple, Set
from collections import defaultdict
import itertools

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, TreeSearch, PC, GES, BIC, BayesianEstimator, ExpertKnowledge
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx

import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

import warnings
import logging
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='pgmpy')
logging.getLogger("pgmpy").setLevel(logging.ERROR)
logging.getLogger("pgmpy.models").setLevel(logging.ERROR)


class Client:
    """
    This class contains the following functionalities required by the clients:
    - building model based on data
    - computing CPTs from a model
    - fusing local and global networks
    - performing K-fold cross validation for global model evaluation on local dataset
    """
    
    def __init__(self) -> None:
        """
        Initialize a Client instance.
        
        Attributes:
            model: The Bayesian network model (initially None)
            dataset: The dataset used for training (initially None)  
            client_size: Size of the client dataset (initially 0)
        """
        self.model: Optional[DiscreteBayesianNetwork] = None
        self.dataset: Optional[pd.DataFrame] = None
        self.client_size: int = 0

    def visualize_network(self,
                structure: Optional[DiscreteBayesianNetwork] = None,
                target_node: str = 'class',
                target_color: str = '#9f0000',
                node_color: str = '#22666F',
                target_size: int = 1200,
                node_size: int = 800,
                figsize: Tuple[int, int] = (10, 8),
                save_path: Optional[str] = None,
                title_prefix: str = "Bayesian Network",
                show_weights: bool = True,
                weight_threshold: float = 0.1) -> None:
        """
        Visualize a (directed) Bayesian network structure using NetworkX and Matplotlib.
        Shows weighted edges with thickness only (no colors) if CPDs are available.
        
        Args:
            structure: The Bayesian network model
            target_node: Node to highlight (default 'class')
            target_color: Color for target node
            node_color: Color for other nodes
            target_size: Size for target node
            node_size: Size for other nodes
            seed: Random seed for layout
            figsize: Figure size
            save_path: Path to save visualization
            title_prefix: Prefix for plot title
            show_weights: Whether to show edge weights when available
            weight_threshold: Minimum weight to display edge labels
        """
        graph = nx.DiGraph()
        
        if structure is not None:
            graph.add_edges_from(structure.edges())
        
        if graph.number_of_nodes() == 0:
            print("No nodes available for visualization.")
            return 
        
        _, ax = plt.subplots(figsize=figsize)
        
        pos = nx.circular_layout(graph)
        
        node_colors = [target_color if n == target_node else node_color for n in graph.nodes()]
        node_sizes  = [target_size if n == target_node else node_size  for n in graph.nodes()]
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
        nx.draw_networkx_labels(graph, pos,
                            labels={n: n for n in graph.nodes()},
                            font_color='black', font_weight='bold', ax=ax)
        
        has_cpds = False
        edge_weights = {}
        
        if structure is not None:
            try:
                cpds = structure.get_cpds()
                has_cpds = len(cpds) > 0
                
                if has_cpds and show_weights:
                    edge_weights = self._calculate_edge_weights(structure)
                    
            except (AttributeError, Exception):
                has_cpds = False
        
        if has_cpds and show_weights and edge_weights:
            self._draw_weighted_edges_thickness_only(graph, pos, edge_weights, weight_threshold, ax)
            title_suffix = " (Weighted Edges - Thickness)"
        else:
            nx.draw_networkx_edges(
                graph, pos,
                arrows=True,                     
                arrowstyle='-|>',               
                arrowsize=20,                   
                edge_color='#d58303',
                width=1.5,
                ax=ax
            )
            title_suffix = " (Simple Edges)"
        
        base_title = f"No. of nodes: {graph.number_of_nodes()}, No. of edges: {graph.number_of_edges()}"
        if has_cpds and show_weights:
            base_title += f", Avg. weight: {np.mean(list(edge_weights.values())):.3f}"
        
        ax.set_title(f"{title_prefix}{title_suffix}\n{base_title}", fontsize=14, pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to: {save_path}")
        
        plt.show()

    def _draw_weighted_edges_thickness_only(self, graph: nx.DiGraph, pos: dict, 
                                        edge_weights: Dict[Tuple[str, str], float], 
                                        weight_threshold: float, ax) -> None:
        """
        Draw edges with weights represented only by line thickness.
        
        Args:
            graph: NetworkX directed graph
            pos: Node positions
            edge_weights: Dictionary of edge weights
            weight_threshold: Minimum weight to show labels
            ax: Matplotlib axis
        """
        edge_widths = {}
        edge_labels = {}
        
        min_width = 0.5
        max_width = 4.0
        
        if edge_weights:
            min_weight = min(edge_weights.values())
            max_weight = max(edge_weights.values())
            weight_range = max_weight - min_weight
            
            for edge in graph.edges():
                weight = edge_weights.get(edge, 0.3)
                
                if weight_range > 0:
                    normalized_weight = (weight - min_weight) / weight_range
                else:
                    normalized_weight = 0.5
                
                edge_widths[edge] = min_width + normalized_weight * (max_width - min_width)
                
                if weight >= weight_threshold:
                    edge_labels[edge] = f'{weight:.2f}'
        else:
            for edge in graph.edges():
                edge_widths[edge] = 1.5
        
        edge_list = list(graph.edges())
        edge_width_list = [edge_widths[edge] for edge in edge_list]
        
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=edge_list,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            edge_color='#d58303',
            width=edge_width_list,
            alpha=0.8,
            ax=ax
        )
        
        if edge_labels:
            label_pos = {}
            for edge, label in edge_labels.items():
                x1, y1 = pos[edge[0]]
                x2, y2 = pos[edge[1]]
                label_x = x1 + 0.6 * (x2 - x1) + 0.02
                label_y = y1 + 0.6 * (y2 - y1) + 0.02
                label_pos[edge] = (label_x, label_y)
            
            for edge, label in edge_labels.items():
                x, y = label_pos[edge]
                ax.text(x, y, label, 
                    fontsize=8, 
                    fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', 
                            facecolor='white', 
                            edgecolor='#404040', 
                            alpha=0.8))
        
        if edge_weights:
            legend_elements = []
            
            thick_line = plt.Line2D([0], [0], color='#404040', linewidth=max_width, 
                                label=f'Strong (≥{max(edge_weights.values()):.2f})')
            medium_line = plt.Line2D([0], [0], color='#404040', linewidth=(min_width + max_width)/2, 
                                    label=f'Medium (~{np.mean(list(edge_weights.values())):.2f})')
            thin_line = plt.Line2D([0], [0], color='#404040', linewidth=min_width, 
                                label=f'Weak (≤{min(edge_weights.values()):.2f})')
            
            legend_elements = [thick_line, medium_line, thin_line]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                    title='Edge Weights')

    def _calculate_edge_weights(self, model: DiscreteBayesianNetwork) -> Dict[Tuple[str, str], float]:
        """
        Calculate edge weights based on dependency strength from CPDs.
        
        Args:
            model: Fitted Bayesian network with CPDs
            
        Returns:
            Dictionary mapping (parent, child) tuples to weight values
        """
        edge_weights = {}
        
        try:
            for cpd in model.get_cpds():
                child = cpd.variable
                parents = cpd.get_evidence() or []
                
                for parent in parents:
                    weight = self._calculate_dependency_strength_from_cpd(parent, child, cpd)
                    edge_weights[(parent, child)] = weight
                    
        except Exception as e:
            print(f"Warning: Error calculating edge weights: {e}")
            for edge in model.edges():
                edge_weights[edge] = 0.5
        
        return edge_weights

    def _calculate_dependency_strength_from_cpd(self, parent: str, child: str, cpd) -> float:
        """Use mutual information instead of TV distance for better sensitivity"""
        try:
            values = np.array(cpd.get_values())
            if values.ndim != 2:
                return 0.0
                
            # Calculate mutual information
            joint_prob = values / np.sum(values)
            parent_marginal = np.sum(joint_prob, axis=0)
            child_marginal = np.sum(joint_prob, axis=1)
            
            mi = 0.0
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if joint_prob[i,j] > 1e-10:
                        mi += joint_prob[i,j] * np.log2(
                            joint_prob[i,j] / (child_marginal[i] * parent_marginal[j] + 1e-10)
                        )
            
            child_entropy = -np.sum(child_marginal * np.log2(child_marginal + 1e-10))
            return mi / (child_entropy + 1e-10) if child_entropy > 0 else 0
            
        except Exception:
            return 0.0

    def create_client_model(self, dataset: pd.DataFrame) -> DiscreteBayesianNetwork:
        df = dataset.dropna().reset_index(drop=True).copy()
        for c in df.columns:
            if not pd.api.types.is_categorical_dtype(df[c]):
                df[c] = df[c].astype("category")
        
        forbidden_edges = []
        if 'class' in df.columns:
            for col in df.columns:
                if col != 'class':
                    forbidden_edges.append(('class', col))
            print(f"Forbidden edges: class cannot be parent of any variable")

        class_constraint = ExpertKnowledge(forbidden_edges=forbidden_edges)

        candidates = {}
        
        try:
            pc = PC(df)
            candidates["pc"] = pc.estimate(variant="stable", ci_test="chi_square", return_type="dag", expert_knowledge=class_constraint, show_progress=False)
            print("PC algorithm completed successfully")
        except Exception as e:
            print(f"PC algorithm failed: {e}")
            pass
        
        try:
            hc = HillClimbSearch(df)
            candidates["hill_climb"] = hc.estimate(scoring_method='bic-d', expert_knowledge=class_constraint, show_progress=False)
            print("Hill Climbing algorithm completed successfully")
        except Exception as e:
            print(f"Hill Climbing algorithm failed: {e}")
            pass
        
        # try:
        #     tree = TreeSearch(df)
        #     candidates["chow_liu"] = tree.estimate(estimator_type="chow-liu", show_progress=False)
        #     print("Chow-Liu Tree algorithm completed successfully")
        # except Exception as e:
        #     print(f"Chow-Liu Tree algorithm failed: {e}")
        #     pass
    
        try:
            ges = GES(df)
            candidates["ges"] = ges.estimate(scoring_method='bic-d', expert_knowledge=class_constraint, debug=False)
            print("GES (Greedy Equivalence Search) algorithm completed successfully")
        except Exception as e:
            print(f"GES algorithm failed: {e}")
            pass
        
        unique_structures = {}
        for name, structure in candidates.items():
            if nx.is_directed_acyclic_graph(structure):
                edge_key = frozenset(structure.edges())
                if edge_key not in unique_structures:
                    unique_structures[edge_key] = (name, structure)
                else:
                    print(f"{name} produced duplicate structure (same as {unique_structures[edge_key][0]})")
        
        print(f"\nUnique valid structures found: {len(unique_structures)}")
        
        if not unique_structures:
            print("No valid structures found - creating empty Bayesian Network")
            empty_bn = DiscreteBayesianNetwork()
            empty_bn.add_nodes_from(df.columns)
            return empty_bn.fit(df, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=1)
        
        bic_scorer = BIC(df)
        best_score = float('-inf')
        best_structure = None
        best_algorithm = None
        
        print("\nEvaluating structures with BIC scoring:")
        print("-" * 50)
        
        for name, structure in unique_structures.values():
            try:
                bn = DiscreteBayesianNetwork()
                bn.add_nodes_from(structure.nodes())
                bn.add_edges_from(structure.edges())
                score = bic_scorer.score(bn)
                
                print(f"{name:15s}: BIC = {score:8.2f} | Edges = {len(structure.edges()):2d} | Nodes = {len(structure.nodes()):2d}")
                
                if score > best_score:
                    best_score = score
                    best_structure = bn
                    best_algorithm = name
            except Exception as e:
                print(f"{name:15s}: FAILED - {e}")
                continue
        
        print("-" * 50)
        
        if best_structure is None:
            print("No structure could be evaluated - creating empty Bayesian Network")
            empty_bn = DiscreteBayesianNetwork()
            empty_bn.add_nodes_from(df.columns)
            return empty_bn.fit(df, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=1)
        
        print(f"BEST ALGORITHM: {best_algorithm.upper()}")
        print(f"Best BIC Score: {best_score:.2f}")
        print(f"Network Structure: {len(best_structure.nodes())} nodes, {len(best_structure.edges())} edges")
        
        if len(best_structure.edges()) > 0:
            print(f"   Edges: {list(best_structure.edges())}")
        
        best_model = best_structure.fit(df, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=1)
        return best_model

    def compute_local_cpts_from_structure_and_data(self, 
                                                  structure: DiscreteBayesianNetwork, 
                                                  data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Compute conditional probability tables (CPTs) from network structure and data.
        
        Args:
            structure: The Bayesian network structure
            data: Training data as a pandas DataFrame
            
        Returns:
            A list of dictionaries containing CPT information for each variable
        """
        model = DiscreteBayesianNetwork(structure)
        model = model.fit(data, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=1)

        cpts_list: List[Dict[str, Any]] = []
        for cpt in model.get_cpds():
            cpt_dict = {
                "variable": cpt.variable,
                "variable_card": cpt.variable_card,
                "values": cpt.values.tolist(),  
                "evidence": cpt.get_evidence(),
                "evidence_card": cpt.cardinality[1:],
                "cardinality": cpt.cardinality.tolist()
            }
            cpts_list.append(cpt_dict)

        return cpts_list
    
    def fuse_bayesian_networks(self, expert_bn, data_bn, data_df,
                                expert_weight=0.7,
                                add_node_threshold=0.6,
                                add_edge_threshold=0.6,
                                reverse_edge_threshold=0.7,
                                remove_edge_threshold=0.3,
                                max_changes=5):
        """
        Fuse expert and data networks by starting with expert network and selectively:
        1. Adding important nodes/edges from data network
        2. Reversing edges if data strongly suggests opposite direction
        3. Removing edges if data shows they are weak
        4. Computing fresh CPDs from data on the final structure
        
        Args:
            expert_bn: Expert knowledge network (foundation)
            data_bn: Data-driven network (modifications source)
            data_df: Training data
            expert_weight: Weight for expert knowledge (0-1) - affects all operations
            add_node_threshold: Minimum strength to add data-only nodes
            add_edge_threshold: Minimum strength to add data-only edges  
            reverse_edge_threshold: Minimum strength to reverse edge direction
            remove_edge_threshold: Maximum strength to keep expert edges (remove if below)
            max_changes: Maximum number of structural changes allowed (default: 5)
        
        Returns:
            Fused Bayesian network with structure and fresh CPDs computed from data
        """       
        expert_weight = max(0.0, min(1.0, expert_weight))
        data_weight = 1.0 - expert_weight
                
        fused_structure = DiscreteBayesianNetwork()
        fused_structure.add_nodes_from(expert_bn.nodes())
        fused_structure.add_edges_from(expert_bn.edges())
        
        expert_nodes = set(expert_bn.nodes())
        expert_edges = set(expert_bn.edges())
        data_nodes = set(data_bn.nodes())
        data_edges = set(data_bn.edges())
                
        changes_made = 0
        
        potential_changes = []
        
        data_only_nodes = data_nodes - expert_nodes
        
        if data_only_nodes:            
            for node in data_only_nodes:
                if node in data_df.columns: 
                    node_importance = self.measure_node_importance(node, data_bn, data_df)
                    
                    weighted_threshold = add_node_threshold + (expert_weight * 0.2)
                    
                    if node_importance >= weighted_threshold:
                        potential_changes.append(('add_node', node, node_importance, weighted_threshold))
        
        current_nodes = set(fused_structure.nodes())
        data_edges_in_scope = {(u, v) for (u, v) in data_edges 
                            if u in current_nodes and v in current_nodes and u in data_df.columns and v in data_df.columns}
        data_only_edges = data_edges_in_scope - expert_edges
        
        if data_only_edges:            
            for u, v in data_only_edges:
                strength = self.measure_edge_strength_from_data(u, v, data_bn, data_df)
                
                weighted_threshold = add_edge_threshold + (expert_weight * 0.2)
                
                if strength >= weighted_threshold:
                    if not self.cyclicity_check(u, v, fused_structure):
                        potential_changes.append(('add_edge', (u, v), strength, weighted_threshold))
                
        current_edges = set(fused_structure.edges())
        
        for u, v in current_edges.intersection(expert_edges):
            if u in current_nodes and v in current_nodes and u in data_df.columns and v in data_df.columns:
                reverse_edge = (v, u)
                
                if reverse_edge in data_edges:
                    forward_strength = self.measure_edge_strength_from_data(u, v, data_bn, data_df)
                    reverse_strength = self.measure_edge_strength_from_data(v, u, data_bn, data_df)
                    
                    strength_diff = reverse_strength - forward_strength
                    
                    weighted_threshold = reverse_edge_threshold + (expert_weight * 0.3)
                    
                    if strength_diff >= weighted_threshold:
                        if not self.cyclicity_check(v, u, fused_structure, exclude_edge=(u, v)):
                            potential_changes.append(('reverse_edge', (u, v), strength_diff, weighted_threshold))
                
        for u, v in current_edges:
            if ((u, v) in expert_edges or (v, u) in expert_edges) and u in data_df.columns and v in data_df.columns:
                data_support = self.measure_edge_support_in_data(u, v, data_bn, data_df)
                
                weighted_threshold = remove_edge_threshold * (1.0 - expert_weight * 0.5)
                
                if data_support <= weighted_threshold:
                    removal_score = weighted_threshold - data_support
                    potential_changes.append(('remove_edge', (u, v), removal_score, weighted_threshold))
        
        potential_changes.sort(key=lambda x: x[2], reverse=True)
               
        for change_type, change_data, score, threshold in potential_changes:
            if changes_made >= max_changes:
                break
                
            if change_type == 'add_node':
                node = change_data
                fused_structure.add_node(node)
                changes_made += 1
                
            elif change_type == 'add_edge':
                u, v = change_data
                if not self.cyclicity_check(u, v, fused_structure):
                    fused_structure.add_edge(u, v)
                    changes_made += 1
                else:
                    print(f"Skipped edge {u}->{v}: would create cycle with current structure")
                    
            elif change_type == 'reverse_edge':
                u, v = change_data
                if not self.cyclicity_check(v, u, fused_structure, exclude_edge=(u, v)):
                    fused_structure.remove_edge(u, v)
                    fused_structure.add_edge(v, u)
                    changes_made += 1
                else:
                    print(f"Skipped reversing {u}->{v}: would create cycle with current structure")
                    
            elif change_type == 'remove_edge':
                u, v = change_data
                fused_structure.remove_edge(u, v)
                changes_made += 1
        
        available_nodes = [node for node in fused_structure.nodes() if node in data_df.columns]
        filtered_data = data_df[available_nodes].dropna().reset_index(drop=True)
        
        if len(filtered_data) == 0:
            print("Warning: No data available for CPD computation")
            return fused_structure
        
        for node in available_nodes:
            if not pd.api.types.is_categorical_dtype(filtered_data[node]):
                filtered_data[node] = filtered_data[node].astype("category")
        
        try:
            fused_bn_with_cpds = fused_structure.fit(filtered_data, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=1)
            
            print(f"Successfully computed CPDs for {len(fused_bn_with_cpds.get_cpds())} variables")
            
            if fused_bn_with_cpds.check_model():
                print("Final network validation successful")
            else:
                print("Warning: Final network validation failed")
                
            return fused_bn_with_cpds
            
        except Exception as e:
            print(f"Error computing CPDs: {e}")
            print("Returning structure-only network")
            return fused_structure


    def measure_node_importance(self, node, data_bn, data_df):
        """
        Measure importance of a node based on connectivity and information content
        """
        try:
            in_degree = data_bn.in_degree(node)
            out_degree = data_bn.out_degree(node)
            total_possible_connections = len(data_bn.nodes()) - 1
            connectivity_score = (in_degree + out_degree) / max(1, total_possible_connections)
            
            if node in data_df.columns:
                node_values = data_df[node].dropna()
                if len(node_values) > 0:
                    value_counts = node_values.value_counts(normalize=True)
                    entropy = -sum(p * np.log2(p + 1e-10) for p in value_counts)
                    max_entropy = np.log2(len(value_counts))
                    entropy_score = entropy / (max_entropy + 1e-10) if max_entropy > 0 else 0
                else:
                    entropy_score = 0
            else:
                entropy_score = 0
            
            importance = 0.6 * connectivity_score + 0.4 * entropy_score
            return min(1.0, importance)
            
        except Exception as e:
            print(f"Error measuring importance for {node}: {e}")
            return 0.0


    def measure_edge_strength_from_data(self, parent, child, network, data_df):
        """
        Measure edge strength based on data evidence - FIXED to use cardinality[1:]
        """
        try:
            if child in network.nodes():
                try:
                    child_cpd = network.get_cpds(child)
                    if child_cpd and parent in (child_cpd.get_evidence() or []):
                        cpt_dict = {
                            'variable': child,
                            'evidence': list(child_cpd.get_evidence() or []),
                            'values': child_cpd.get_values(),
                            'cardinality': list(child_cpd.cardinality)  
                        }
                        return self.measure_dependency_strength(parent, child, cpt_dict)
                except:
                    pass
            
            if parent in data_df.columns and child in data_df.columns:
                return self.compute_mutual_information(parent, child, data_df)
            
            return 0.0
        
        except Exception as e:
            print(f"Error measuring edge strength {parent}->{child}: {e}")
            return 0.0
    
    def measure_edge_support_in_data(self, parent, child, data_bn, data_df):
        """
        Measure how well an edge is supported by data evidence
        """
        try:
            if data_bn.has_edge(parent, child):
                return self.measure_edge_strength_from_data(parent, child, data_bn, data_df)
            
            if parent in data_df.columns and child in data_df.columns:
                return self.compute_mutual_information(parent, child, data_df)
            
            return 0.0
            
        except Exception as e:
            print(f"Error measuring edge support {parent}->{child}: {e}")
            return 0.0


    def compute_mutual_information(self, var1, var2, data_df):
        """
        Compute normalized mutual information between two variables from data
        """
        try:
            df_clean = data_df[[var1, var2]].dropna()
            if len(df_clean) < 10:  
                return 0.0
                
            x_values = df_clean[var1].astype('category').cat.codes
            y_values = df_clean[var2].astype('category').cat.codes
            
            joint_counts = pd.crosstab(x_values, y_values)
            joint_probs = joint_counts / joint_counts.sum().sum()
            
            x_probs = joint_probs.sum(axis=1)
            y_probs = joint_probs.sum(axis=0)
            
            mi = 0.0
            for i in joint_probs.index:
                for j in joint_probs.columns:
                    if joint_probs.loc[i, j] > 0:
                        mi += joint_probs.loc[i, j] * np.log2(
                            joint_probs.loc[i, j] / (x_probs[i] * y_probs[j] + 1e-10)
                        )
            
            y_entropy = -sum(p * np.log2(p + 1e-10) for p in y_probs if p > 0)
            normalized_mi = mi / (y_entropy + 1e-10) if y_entropy > 0 else 0
            
            return max(0.0, min(1.0, normalized_mi))
            
        except Exception as e:
            print(f"Error computing mutual information between {var1} and {var2}: {e}")
            return 0.0


    def cyclicity_check(self, u, v, network, exclude_edge=None):
        """
        Check if adding edge u->v would create a cycle
        """
        try:
            temp_graph = nx.DiGraph()
            temp_graph.add_nodes_from(network.nodes())
            
            for edge in network.edges():
                if exclude_edge is None or edge != exclude_edge:
                    temp_graph.add_edge(*edge)
            
            temp_graph.add_edge(u, v)
            
            return not nx.is_directed_acyclic_graph(temp_graph)
            
        except Exception as e:
            print(f"Error checking cycle for edge {u}->{v}: {e}")
            return True  


    def get_inference_path(self, model, evidence, target):
        """Find paths from evidence to target through network"""
        target_parents = list(model.predecessors(target))
        evidence_vars = list(evidence.keys())
        
        paths = []
        for ev_var in evidence_vars:
            if ev_var in target_parents:
                paths.append(f"{ev_var}->{target}")
            else:
                for parent in target_parents:
                    if nx.has_path(model.to_undirected(), ev_var, parent):
                        try:
                            path = nx.shortest_path(model.to_undirected(), ev_var, parent)
                            paths.append(f"{'->'.join(path)}->{target}")
                        except:
                            continue
        
        if target_parents:
            target_deps = f"P({target}|{','.join(target_parents)})"
        else:
            target_deps = f"P({target})"
        
        path_summary = f"{target_deps}; " + "; ".join(paths[:3])  
        return path_summary if path_summary != f"{target_deps}; " else target_deps


    def kfold_cv(self, data, model, target='class', k=3, csv_filename=None):
        """
        Strict K-Fold cross-validation for Bayesian Networks - NO FALLBACKS
        
        Args:
            data: pandas DataFrame
            model: pgmpy BayesianNetwork 
            target: target variable name
            k: number of folds
            csv_filename: optional CSV file to save results
            
        Returns:
            dict with accuracy and detailed statistics
        """
        
        data = data.copy()
        data[target] = data[target].astype(int)
        
        data['predicted_class'] = np.nan
        data['prediction_confidence'] = np.nan
        data['all_class_probabilities'] = ''
        data['inference_path'] = '' 
        data['prediction_status'] = ''  
        
        skf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        fold_accuracies = []
        fold_stats = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(data.drop(columns=[target]), data[target]), 1):
            print(f"\n--- Fold {fold}/{k} ---")
            
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            successful_predictions = 0
            failed_predictions = 0
            correct_predictions = 0
            
            try:
                model.fit(train_data, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=1)
                infer = VariableElimination(model)
                target_cpd = model.get_cpds(target)
                target_states = (target_cpd.state_names[target] 
                            if hasattr(target_cpd, 'state_names') 
                            else list(range(target_cpd.variable_card)))
                
                print(f"Model fitted successfully.")
                
            except Exception as e:
                print(f"CRITICAL: Model fitting failed for fold {fold}: {e}")
                for i in test_idx:
                    data.at[i, 'prediction_status'] = 'Model Fitting Failed'
                    data.at[i, 'inference_path'] = ''
                fold_stats.append({
                    'fold': fold,
                    'total_samples': len(test_idx),
                    'successful_predictions': 0,
                    'failed_predictions': len(test_idx),
                    'accuracy_on_successful': np.nan,
                    'overall_accuracy': 0.0
                })
                fold_accuracies.append(0.0)
                continue
            
            for i in test_idx:
                try:
                    evidence = {}
                    for node in model.nodes():
                        if node != target and node in data.columns:
                            val = int(data.iloc[i][node])
                            
                            if val in train_data[node].values:
                                evidence[node] = val
                            else:
                                raise ValueError(f"Unseen value {val} for feature {node}")
   
                    query_result = infer.query(variables=[target], evidence=evidence, show_progress=False)
                    
                    prob_dist = query_result.values.flatten()
                    prob_dist = prob_dist / prob_dist.sum()  
                    pred_idx = np.argmax(prob_dist)
                    if isinstance(target_states[0], str):
                        pred = int(target_states[pred_idx])
                    else:
                        pred = target_states[pred_idx]
                    
                    confidence = float(prob_dist[pred_idx])
                    prob_str = ','.join([f'{prob:.4f}' for prob in prob_dist])
                    
                    inference_path = self.get_inference_path(model, evidence, target)
                    
                    data.at[i, 'predicted_class'] = int(pred)
                    data.at[i, 'prediction_confidence'] = confidence
                    data.at[i, 'all_class_probabilities'] = prob_str
                    data.at[i, 'inference_path'] = inference_path
                    data.at[i, 'prediction_status'] = 'Success'
                    
                    successful_predictions += 1
                    
                    if pred == data.iloc[i][target]:
                        correct_predictions += 1
                        
                except Exception as e:
                    data.at[i, 'predicted_class'] = np.nan
                    data.at[i, 'prediction_confidence'] = np.nan
                    data.at[i, 'all_class_probabilities'] = ''
                    data.at[i, 'inference_path'] = ''
                    data.at[i, 'prediction_status'] = f'Inference Failed: {str(e)[:50]}'
                    
                    failed_predictions += 1
            
            # Calculate fold statistics
            total_samples = len(test_idx)
            
            accuracy_on_successful = (correct_predictions / successful_predictions 
                                    if successful_predictions > 0 else np.nan)

            overall_accuracy = correct_predictions / total_samples
            
            fold_stats.append({
                'fold': fold,
                'total_samples': total_samples,
                'successful_predictions': successful_predictions,
                'failed_predictions': failed_predictions,
                'correct_predictions': correct_predictions,
                'accuracy_on_successful': accuracy_on_successful,
                'overall_accuracy': overall_accuracy,
                'success_rate': successful_predictions / total_samples
            })
            
            fold_accuracies.append(overall_accuracy)
            
            print(f"Fold {fold} Results:")
            print(f"  Total samples: {total_samples}")
            print(f"  Successful predictions: {successful_predictions}")
            print(f"  Failed predictions: {failed_predictions}")
            print(f"  Correct predictions: {correct_predictions}")
            print(f"  Accuracy on successful: {accuracy_on_successful:.4f}" if not np.isnan(accuracy_on_successful) else "  Accuracy on successful: N/A")
            print(f"  Overall accuracy: {overall_accuracy:.4f}")
        
        avg_accuracy = np.mean(fold_accuracies)
        successful_samples = sum(stats['successful_predictions'] for stats in fold_stats)
        total_samples = sum(stats['total_samples'] for stats in fold_stats)
        overall_success_rate = successful_samples / total_samples
        
        successful_correct = sum(stats['correct_predictions'] for stats in fold_stats)
        accuracy_on_successful_only = successful_correct / successful_samples if successful_samples > 0 else np.nan
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Overall Accuracy (failed = wrong): {avg_accuracy:.4f}")
        print(f"Accuracy on successful predictions only: {accuracy_on_successful_only:.4f}" if not np.isnan(accuracy_on_successful_only) else "Accuracy on successful predictions only: N/A")
        print(f"Success rate: {overall_success_rate:.4f} ({successful_samples}/{total_samples})")
        print(f"Failed predictions: {total_samples - successful_samples}")
        
        if csv_filename:
            data['predicted_class'] = data['predicted_class'].astype('Int64') 
            data.to_csv(csv_filename, index=False)
            print(f"Detailed results saved to: {csv_filename}")
        
        return {
            'average_accuracy': round(avg_accuracy, 4),
            'accuracy_on_successful_only': round(accuracy_on_successful_only, 4) if not np.isnan(accuracy_on_successful_only) else None,
            'success_rate': round(overall_success_rate, 4),
            'total_samples': total_samples,
            'successful_predictions': successful_samples,
            'failed_predictions': total_samples - successful_samples,
            'fold_details': fold_stats
        }

class Coordinator(Client):
    """
    This class extends the Client class for federated Bayesian network learning. 

    It contains the following functionalities required by the coordinator (server):
    - creating an expert knowledge-based initial network
    - aggregating clients CPTs 
    - building a model from the aggregated CPTs
    - comparing two CPTs list for convergence check
    """
    
    def majority_voted_network(self, 
                              whitelists: List[List[List[str]]],
                              blacklists: List[List[List[str]]],
                              tie_breaker: str = "exclude") -> Set[Tuple[str, str]]:
        """
        Perform majority voting on edge lists to determine final network structure.
        
        Args:
            whitelists: List of whitelisted edge lists from each client
            blacklists: List of blacklisted edge lists from each client
            tie_breaker: Strategy for handling ties ("include" or "exclude")
            
        Returns:
            Set of edges (as tuples) that received majority votes
        """
        all_edges: Set[Tuple[str, str]] = set()

        def add_valid_edges(edge_list: List[List[str]]) -> None:
            for edge in edge_list:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    u, v = edge
                    if isinstance(u, str) and isinstance(v, str):
                        all_edges.add((u, v))

        for wl in whitelists:
            add_valid_edges(wl)
        for bl in blacklists:
            add_valid_edges(bl)

        wl_votes: Dict[Tuple[str, str], Set[int]] = {edge: set() for edge in all_edges}
        bl_votes: Dict[Tuple[str, str], Set[int]] = {edge: set() for edge in all_edges}

        for client_idx, wl in enumerate(whitelists):
            for edge in wl:
                if tuple(edge) in wl_votes:
                    wl_votes[tuple(edge)].add(client_idx)

        for client_idx, bl in enumerate(blacklists):
            for edge in bl:
                if tuple(edge) in bl_votes:
                    bl_votes[tuple(edge)].add(client_idx)

        final_edges: Set[Tuple[str, str]] = set()

        for edge in all_edges:
            wl_count = len(wl_votes[edge])
            bl_count = len(bl_votes[edge])

            if wl_count > bl_count:
                final_edges.add(edge)
            elif bl_count > wl_count:
                continue 
            else:  # tie condition
                if tie_breaker == "include":
                    final_edges.add(edge)

        return final_edges


    def create_constrained_global_network(self,
                                        whitelists: List[List[List[str]]],
                                        blacklists: List[List[List[str]]],
                                        tie_breaker: str = "exclude") -> DiscreteBayesianNetwork:
        """
        Create a global network structure using majority voting with cycle detection.
        
        Args:
            whitelists: List of whitelisted edge lists from each client
            blacklists: List of blacklisted edge lists from each client
            tie_breaker: Strategy for handling ties ("include" or "exclude")
            
        Returns:
            DiscreteBayesianNetwork with the voted structure
        """
        # Majority voting
        final_edges = self.majority_voted_network(
            whitelists,
            blacklists,
            tie_breaker
        )

        valid_edges = [
            (str(u), str(v))
            for (u, v) in final_edges
            if isinstance(u, str) and isinstance(v, str) and len((u, v)) == 2
        ]

        model = DiscreteBayesianNetwork()

        if valid_edges:
            nodes = {u for u, v in valid_edges} | {v for u, v in valid_edges}
            for node in nodes:
                model.add_node(node)

            G = nx.DiGraph()
            G.add_nodes_from(nodes)

            for edge in valid_edges:
                G.add_edge(*edge)
                if nx.is_directed_acyclic_graph(G):
                    model.add_edge(*edge)
                else:
                    G.remove_edge(*edge)  

        else:
            all_variables = set()

            for wl in whitelists:
                for edge in wl:
                    if isinstance(edge, (list, tuple)) and len(edge) == 2:
                        u, v = edge
                        if isinstance(u, str) and isinstance(v, str):
                            all_variables.add(u)
                            all_variables.add(v)

            for bl in blacklists:
                for edge in bl:
                    if isinstance(edge, (list, tuple)) and len(edge) == 2:
                        u, v = edge
                        if isinstance(u, str) and isinstance(v, str):
                            all_variables.add(u)
                            all_variables.add(v)

            for variable in all_variables:
                model.add_node(variable)

        return model
    
   
    def aggregate_cpts(self, clients_cpts: List[List[Dict[str, Any]]], dataset_sizes: List[float]):
        """
        Aggregate CPTs with enhanced numerical stability checks.
        """
        if len(clients_cpts) != len(dataset_sizes):
            raise ValueError("Number of clients and dataset_sizes must match.")
        
        total = float(sum(dataset_sizes))
        if total <= 0:
            raise ValueError("Sum of dataset sizes must be > 0.")
        weights = [s / total for s in dataset_sizes]

        by_var: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for ci, cpt_list in enumerate(clients_cpts):
            for cpd in cpt_list:
                by_var[cpd['variable']].append((ci, cpd))

        aggregated = []

        for var, entries in by_var.items():
            try:
                groups = defaultdict(list)
                for ci, cpd in entries:
                    groups[int(cpd['variable_card'])].append((ci, cpd))
                best_var_card = max(groups.keys(), key=lambda k: sum(weights[ci] for ci, _ in groups[k]))
                chosen = groups[best_var_card]
                var_card = best_var_card

                parent_to_cards: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
                for ci, cpd in chosen:
                    ev_raw = cpd.get('evidence')
                    ev = list(ev_raw) if ev_raw is not None else []
                    evc_raw = cpd.get('evidence_card')
                    evc = list(evc_raw) if evc_raw is not None else []
                    
                    if len(ev) != len(evc):
                        print(f"Warning: Evidence length mismatch for {var}, client {ci}")
                        continue
                        
                    ev_map = dict(zip(ev, map(int, evc)))
                    
                    for p, c in ev_map.items():
                        if c > 0:  
                            parent_to_cards[p][int(c)] += weights[ci]

                sup_parents = sorted(parent_to_cards.keys())
                sup_parent_cards = {}
                for p in sup_parents:
                    choices = parent_to_cards[p]
                    if choices:
                        chosen_card = max(choices.keys(), key=lambda card: choices[card])
                        sup_parent_cards[p] = max(2, chosen_card)  
                    else:
                        sup_parent_cards[p] = 2
                        
                sup_parent_card_list = [sup_parent_cards[p] for p in sup_parents]
                per_client = []
                for ci, cpd in chosen:
                    try:
                        ev_raw = cpd.get('evidence')
                        ev = list(ev_raw) if ev_raw is not None else []
                        evc_raw = cpd.get('evidence_card')
                        evc_list = list(map(int, list(evc_raw) if evc_raw is not None else []))

                        client_sel_supidx = [sup_parents.index(p) for p in ev if p in sup_parents]
                        varc = int(cpd['variable_card'])
                        values = cpd['values']

                        if values is None:
                            print(f"Warning: None values for {var}, client {ci}")
                            continue
                            
                        per_client.append((ci, values, varc, evc_list, client_sel_supidx))
                    except Exception as e:
                        print(f"Warning: Error processing client {ci} for variable {var}: {e}")
                        continue

                table = {}
                if sup_parent_card_list:
                    try:
                        for idx_sup in itertools.product(*[range(c) for c in sup_parent_card_list]):
                            acc = [0.0] * var_card
                            wsum = 0.0
                            
                            for (ci, values, varc, evc_list, client_sel_supidx) in per_client:
                                try:
                                    subset = []
                                    out_of_range = False
                                    for k, sup_idx in enumerate(client_sel_supidx):
                                        if sup_idx >= len(idx_sup):
                                            out_of_range = True
                                            break
                                        sup_val = idx_sup[sup_idx]
                                        if k >= len(evc_list) or sup_val >= evc_list[k]:
                                            out_of_range = True
                                            break
                                        subset.append(sup_val)
                                        
                                    if out_of_range:
                                        continue

                                    prob_vector = self.extract_probability_vector(
                                        values, var_card, subset, evc_list
                                    )
                                    
                                    if len(prob_vector) != var_card:
                                        continue
                                        
                                    if any(np.isnan(prob_vector)) or any(np.isinf(prob_vector)):
                                        print(f"Warning: Invalid probabilities for {var}, client {ci}")
                                        continue
                                        
                                    w = weights[ci]
                                    wsum += w
                                    for i in range(var_card):
                                        acc[i] += w * max(0.0, float(prob_vector[i]))  
                                        
                                except Exception as e:
                                    print(f"Warning: Error in aggregation for {var}, client {ci}: {e}")
                                    continue

                            if wsum > 1e-10:
                                acc = [x / wsum for x in acc]
                            else:
                                acc = [1.0 / var_card] * var_card
                            
                            s = sum(acc)
                            if s <= 1e-10 or np.isnan(s) or np.isinf(s):
                                normalized_acc = [1.0 / var_card] * var_card
                            else:
                                normalized_acc = [max(1e-10, x / s) for x in acc]
                                
                            final_sum = sum(normalized_acc)
                            if abs(final_sum - 1.0) > 1e-6:
                                normalized_acc = [x / final_sum for x in normalized_acc]
                            
                            table[idx_sup] = normalized_acc
                            
                    except Exception as e:
                        print(f"Warning: Error building table for {var}: {e}")
                        for idx_sup in itertools.product(*[range(c) for c in sup_parent_card_list]):
                            table[idx_sup] = [1.0/var_card] * var_card
                else:
                    table[tuple()] = [1.0/var_card] * var_card

                if not sup_parent_card_list:
                    nested_values = [table[tuple()][s] for s in range(var_card)]
                else:
                    nested_values = []
                    for s in range(var_card):
                        def build_nested(dim: int, prefix: Tuple[int, ...]):
                            if dim == len(sup_parent_card_list):
                                return table.get(prefix, [1.0/var_card] * var_card)[s]
                            return [build_nested(dim+1, prefix + (i,)) for i in range(sup_parent_card_list[dim])]
                        nested_values.append(build_nested(0, tuple()))

                if any(np.isnan(self.flatten_nested(nested_values))) or any(np.isinf(self.flatten_nested(nested_values))):
                    print(f"Warning: Invalid final values for {var}, using uniform distribution")
                    nested_values = [[1.0/var_card] * var_card for _ in range(var_card)]

                aggregated.append({
                    'variable': var,
                    'variable_card': var_card,
                    'values': nested_values,
                    'evidence': sup_parents,
                    'evidence_card': sup_parent_card_list,
                    'cardinality': [var_card] + sup_parent_card_list,
                })
                
            except Exception as e:
                print(f"Error aggregating variable {var}: {e}")
                aggregated.append({
                    'variable': var,
                    'variable_card': 2,  
                    'values': [0.5, 0.5],
                    'evidence': [],
                    'evidence_card': [],
                    'cardinality': [2],
                })

        aggregated.sort(key=lambda d: d['variable'])
        return aggregated

    def extract_probability_vector(self, values, var_card, subset, evc_list):
        """
        Safely extract probability vector from nested values structure.
        """
        try:
            prob_vector = []
            
            if len(evc_list) == 0:
                if isinstance(values, (int, float)):
                    prob_vector = [float(values)] if var_card == 1 else [1.0/var_card] * var_card
                elif hasattr(values, '__len__') and len(values) >= var_card:
                    prob_vector = [float(values[s]) for s in range(var_card)]
                else:
                    prob_vector = [1.0/var_card] * var_card
            else:
                values_array = np.array(values)
                if values_array.ndim == 2:
                    strides = []
                    for i in range(len(evc_list)):
                        stride = 1
                        for j in range(i+1, len(evc_list)):
                            stride *= max(1, evc_list[j])
                        strides.append(stride)
                    
                    col = sum(min(idx, max(0, card-1)) * stride 
                            for idx, stride, card in zip(subset, strides, evc_list))
                    col = min(col, values_array.shape[1] - 1)
                    
                    prob_vector = [max(0.0, float(values_array[s, col])) for s in range(min(var_card, values_array.shape[0]))]
                    
                    while len(prob_vector) < var_card:
                        prob_vector.append(1.0/var_card)
                else:
                    prob_vector = [1.0/var_card] * var_card
                    
            prob_vector = [max(1e-10, min(1.0, p)) for p in prob_vector]
            
            s = sum(prob_vector)
            if s > 1e-10:
                prob_vector = [p/s for p in prob_vector]
            else:
                prob_vector = [1.0/var_card] * var_card
                
            return prob_vector
            
        except Exception as e:
            print(f"Warning: Error extracting probability vector: {e}")
            return [1.0/var_card] * var_card

    def flatten_nested(self, nested_list):
        """Flatten nested list structure for validation."""
        result = []
        if isinstance(nested_list, (list, tuple, np.ndarray)):
            for item in nested_list:
                result.extend(self.flatten_nested(item))
        else:
            result.append(float(nested_list))
        return result

    def get_edge_support_from_clients(self, parent, child, cpt_lists, weights):
        total_weight = sum(weights)
        weighted_support = 0.0
        
        for i, cpts in enumerate(cpt_lists):
            weight = weights[i]
            child_cpt = next((c for c in cpts if c['variable'] == child), None)
            
            if child_cpt and parent in child_cpt.get('evidence', []):
                strength = self.measure_dependency_strength(parent, child, child_cpt)
                weighted_support += weight * strength
            
        return weighted_support / total_weight if total_weight > 0 else 0.0

    def learn_parameters(self, cpt_lists, weights, max_changes=5, 
                     addition_threshold=0.5, removal_threshold=0.2, 
                     reversal_threshold=0.6, node_addition_threshold=0.8,
                     forbidden_edges = None):
        """
        Learn structure using client consensus with separate thresholds for different operations
        
        Args:
            cpt_lists: List of CPT lists from clients
            weights: Client weights for consensus
            max_changes: Maximum structural changes per iteration
            addition_threshold: Minimum consensus for adding edges (0.5 = 50% support)
            removal_threshold: Maximum consensus for removing edges (0.2 = remove if <20% support)
            reversal_threshold: Minimum consensus for reversing edges (0.6 = 60% support)
            node_addition_threshold: Minimum consensus for adding nodes (0.8 = 80% support)
        """
        current_cpts = self.aggregate_cpts(cpt_lists, weights)
        current_variables = set([cpt['variable'] for cpt in current_cpts])
        
        all_client_variables = set()
        for cpts in cpt_lists:
            for cpt in cpts:
                all_client_variables.add(cpt['variable'])
                all_client_variables.update(cpt.get('evidence', []))
        
        candidate_new_variables = all_client_variables - current_variables
 
        current_edges = set()
        for cpt in current_cpts:
            child = cpt['variable']
            for parent in cpt['evidence']:
                current_edges.add((parent, child))
        
        initial_node_count = len(current_variables)
        initial_edge_count = len(current_edges)
      
        changes_made = 0
        total_improvement = 0.0
        
        def get_node_degree(node, edges):
            return sum(1 for p, c in edges if p == node or c == node)
        
        while changes_made < max_changes:
            best_change = None
            best_score = 0.01
            
            for new_var in candidate_new_variables:
                node_score = self.score_node_addition_consensus(
                    new_var, cpt_lists, weights, node_addition_threshold
                )
                
                if node_score > best_score:
                    best_change = ('add_node', new_var, node_score)
                    best_score = node_score
            
            for parent in current_variables:
                for child in current_variables:
                    if parent != child and (parent, child) not in current_edges:

                        # if forbidden_edges and (parent, child) in forbidden_edges:
                        #     continue

                        if self.cyclicity_check(parent, child, current_edges):
                            continue
                        
                        child_cpt = next((c for c in current_cpts if c['variable'] == child), None)
                        if child_cpt and len(child_cpt['evidence']) >= 4:  
                            continue
                        
                        edge_score = self.score_edge_addition_consensus(
                            parent, child, cpt_lists, weights, addition_threshold
                        )
                        
                        if edge_score > best_score:
                            memory_cost = self.estimate_edge_addition_cost(parent, child, current_cpts)
                            if memory_cost < 8000:
                                best_change = ('add_edge', (parent, child), edge_score)
                                best_score = edge_score
            
            for parent, child in current_edges:
                parent_degree = get_node_degree(parent, current_edges)
                child_degree = get_node_degree(child, current_edges)
                
                if parent_degree <= 1 or child_degree <= 1:
                    continue
                
                removal_score = self.score_edge_removal_consensus(
                    parent, child, cpt_lists, weights, removal_threshold
                )
                
                if removal_score > best_score:
                    best_change = ('remove_edge', (parent, child), removal_score)
                    best_score = removal_score
            
            for parent, child in current_edges:
                if (child, parent) not in current_edges:
                    reversal_score = self.score_edge_reversal_consensus(
                        parent, child, cpt_lists, weights, reversal_threshold
                    )
                    
                    if reversal_score > best_score:
                        if not self.cyclicity_check(child, parent, current_edges - {(parent, child)}):
                            best_change = ('reverse_edge', (parent, child), reversal_score)
                            best_score = reversal_score
            
            if best_change is None:
                print("No beneficial changes found, stopping")
                break
            
            action, change_data, score = best_change
            
            if action == 'add_node':
                new_var = change_data
                new_cpt = self.create_initial_cpt_for_new_node(new_var, cpt_lists, weights)
                current_cpts.append(new_cpt)
                current_variables.add(new_var)
                candidate_new_variables.remove(new_var)
                print(f"Added node {new_var}, score: {score:.4f}")
                
            elif action == 'add_edge':
                parent, child = change_data
                current_cpts = self.add_edge_to_cpts(parent, child, current_cpts)
                current_edges.add((parent, child))
                print(f"Added edge {parent}->{child}, score: {score:.4f}")
                
            elif action == 'remove_edge':
                parent, child = change_data
                current_cpts = self.remove_edge_from_cpts(parent, child, current_cpts)
                current_edges.remove((parent, child))
                print(f"Removed edge {parent}->{child}, score: {score:.4f}")
                
            elif action == 'reverse_edge':
                parent, child = change_data

                # if forbidden_edges and (child, parent) in forbidden_edges:
                #     continue

                current_cpts = self.remove_edge_from_cpts(parent, child, current_cpts)
                current_cpts = self.add_edge_to_cpts(child, parent, current_cpts)
                current_edges.remove((parent, child))
                current_edges.add((child, parent))
                print(f"Reversed edge {parent}->{child} to {child}->{parent}, score: {score:.4f}")
            
            changes_made += 1
            total_improvement += score
            
            node_degrees = {node: get_node_degree(node, current_edges) for node in current_variables}
            isolated_nodes = [node for node, degree in node_degrees.items() if degree == 0]
            if isolated_nodes:
                print(f"WARNING: Isolated nodes: {isolated_nodes}")
        
        final_node_count = len(current_variables)
        final_edge_count = len(current_edges)
        active_nodes = len([node for node in current_variables if get_node_degree(node, current_edges) > 0])
        
        return current_cpts


    def score_edge_addition_consensus(self, parent, child, cpt_lists, weights, addition_threshold):
        """
        Score edges for addition based on client consensus and dependency strength
        """
        edge_support = self.get_edge_support_from_clients(parent, child, cpt_lists, weights)
        
        if edge_support < addition_threshold:
            return 0.0
        
        avg_strength = 0.0
        supporting_weight = 0.0
        
        for i, cpts in enumerate(cpt_lists):
            weight = weights[i]
            child_cpt = next((c for c in cpts if c['variable'] == child), None)
            
            if child_cpt and parent in child_cpt['evidence']:
                strength = self.measure_dependency_strength(parent, child, child_cpt)
                avg_strength += weight * strength
                supporting_weight += weight
        
        if supporting_weight == 0:
            return 0.0
        
        avg_strength /= supporting_weight
        
        consensus_score = edge_support * avg_strength
        
        if edge_support > (addition_threshold + 0.3):
            consensus_score *= 1.3
        
        return consensus_score


    def score_edge_removal_consensus(self, parent, child, cpt_lists, weights, removal_threshold):
        """
        Score edge removal - higher score means removal is more beneficial
        
        Args:
            removal_threshold: Maximum consensus for removal (0.2 = remove if <20% support)
        """
        edge_support = self.get_edge_support_from_clients(parent, child, cpt_lists, weights)
        
        if edge_support >= removal_threshold:
            return 0.0
        
        avg_weakness = 0.0
        total_weight = 0.0
        
        for i, cpts in enumerate(cpt_lists):
            weight = weights[i]
            child_cpt = next((c for c in cpts if c['variable'] == child), None)
            
            if child_cpt:
                if parent in child_cpt['evidence']:
                    strength = self.measure_dependency_strength(parent, child, child_cpt)
                    weakness = max(0, 0.4 - strength) 
                    avg_weakness += weight * weakness
                else:
                    avg_weakness += weight * 0.3
                
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        avg_weakness /= total_weight
        
        consensus_gap = removal_threshold - edge_support
        removal_score = avg_weakness * consensus_gap * 2.0  
        
        if edge_support > (removal_threshold * 0.7):
            removal_score *= 0.5
        
        return removal_score


    def score_edge_reversal_consensus(self, parent, child, cpt_lists, weights, reversal_threshold):
        """
        Score edge reversal based on client consensus for opposite direction
        """
        current_support = self.get_edge_support_from_clients(parent, child, cpt_lists, weights)
        
        reverse_support = self.get_edge_support_from_clients(child, parent, cpt_lists, weights)
        
        if reverse_support < reversal_threshold:
            return 0.0
        
        if reverse_support <= current_support:
            return 0.0
        
        current_strength = 0.0
        reverse_strength = 0.0
        total_weight = 0.0
        
        for i, cpts in enumerate(cpt_lists):
            weight = weights[i]
            
            child_cpt = next((c for c in cpts if c['variable'] == child), None)
            if child_cpt and parent in child_cpt['evidence']:
                current_strength += weight * self.measure_dependency_strength(parent, child, child_cpt)
            
            parent_cpt = next((c for c in cpts if c['variable'] == parent), None)
            if parent_cpt and child in parent_cpt['evidence']:
                reverse_strength += weight * self.measure_dependency_strength(child, parent, parent_cpt)
            
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        current_strength /= total_weight
        reverse_strength /= total_weight
        
        consensus_improvement = reverse_support - current_support
        strength_improvement = reverse_strength - current_strength
        
        reversal_score = consensus_improvement * (1 + strength_improvement)
        
        return max(0.0, reversal_score)


    def score_node_addition_consensus(self, new_var, cpt_lists, weights, node_addition_threshold):
        """
        Score adding a new node based on how many clients have it and its importance
        """
        clients_with_node = 0
        weighted_importance = 0.0
        total_weight = 0.0
        
        for i, cpts in enumerate(cpt_lists):
            weight = weights[i]
            total_weight += weight
            
            has_variable = any(cpt['variable'] == new_var for cpt in cpts)
            if has_variable:
                clients_with_node += 1
                
                node_importance = self.measure_node_importance_in_client_cpts(new_var, cpts)
                weighted_importance += weight * node_importance
        
        if total_weight == 0:
            return 0.0
        
        consensus = clients_with_node / len(cpt_lists)
        
        if consensus < node_addition_threshold:
            return 0.0
        
        avg_importance = weighted_importance / total_weight if clients_with_node > 0 else 0.0
        
        node_score = consensus * avg_importance
        
        if consensus > (node_addition_threshold + 0.1):
            node_score *= 1.2
        
        return node_score


    def measure_node_importance_in_client_cpts(self, node, cpts):
        """
        Measure importance of a node within a client's CPT list
        """
        connections = 0
        total_possible = len(cpts) - 1
        
        node_cpt = next((cpt for cpt in cpts if cpt['variable'] == node), None)
        if node_cpt:
            connections += len(node_cpt.get('evidence', []))
        
        for cpt in cpts:
            if node in cpt.get('evidence', []):
                connections += 1
        
        connectivity_score = connections / max(1, total_possible)
        
        if node_cpt:
            values = np.array(node_cpt['values'])
            if values.size > 0:
                if values.ndim == 1:
                    probs = values / (np.sum(values) + 1e-10)
                else:
                    probs = np.mean(values, axis=1)
                    probs = probs / (np.sum(probs) + 1e-10)
                
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                max_entropy = np.log2(len(probs))
                entropy_score = entropy / (max_entropy + 1e-10) if max_entropy > 0 else 0
            else:
                entropy_score = 0.5 
        else:
            entropy_score = 0.5 
        
        importance = 0.6 * connectivity_score + 0.4 * entropy_score
        return min(1.0, importance)


    def create_initial_cpt_for_new_node(self, new_var, cpt_lists, weights):
        """
        Create initial CPT for a newly added node by aggregating from clients that have it
        """
        supporting_cpts = []
        supporting_weights = []
        
        for i, cpts in enumerate(cpt_lists):
            weight = weights[i]
            node_cpt = next((cpt for cpt in cpts if cpt['variable'] == new_var), None)
            if node_cpt:
                supporting_cpts.append([node_cpt]) 
                supporting_weights.append(weight)
        
        if not supporting_cpts:
            return {
                'variable': new_var,
                'variable_card': 2,  
                'values': [0.5, 0.5],
                'evidence': [],
                'evidence_card': [],
                'cardinality': [2]
            }
        
        try:
            aggregated = self.aggregate_cpts(supporting_cpts, supporting_weights)
            return aggregated[0] if aggregated else {
                'variable': new_var,
                'variable_card': 2,
                'values': [0.5, 0.5], 
                'evidence': [],
                'evidence_card': [],
                'cardinality': [2]
            }
        except Exception as e:
            print(f"Error aggregating CPT for new node {new_var}: {e}")
            return {
                'variable': new_var,
                'variable_card': 2,
                'values': [0.5, 0.5], 
                'evidence': [],
                'evidence_card': [],
                'cardinality': [2]
            }


    def get_edge_support_from_clients(self, parent, child, cpt_lists, weights):
        """
        Calculate weighted consensus for an edge across clients
        """
        total_weight = sum(weights)
        supporting_weight = 0.0
        
        for i, cpts in enumerate(cpt_lists):
            weight = weights[i]
            
            child_cpt = next((c for c in cpts if c['variable'] == child), None)
            if child_cpt and parent in child_cpt.get('evidence', []):
                supporting_weight += weight
        
        return supporting_weight / total_weight if total_weight > 0 else 0.0


    def measure_dependency_strength(self, parent, child, child_cpt):
        """
        Measure how strongly child depends on parent in this CPT using information gain
        Fixed to handle missing evidence_card gracefully
        """
        try:
            values = np.array(child_cpt['values'])
            evidence = child_cpt.get('evidence', [])
            
            if parent not in evidence:
                return 0.0
            
            parent_idx = evidence.index(parent)
            
            cardinality = child_cpt.get('cardinality', [])
            if len(cardinality) > 1:
                evidence_card = cardinality[1:] 
            else:
                evidence_card = child_cpt.get('evidence_card', [])
                if not evidence_card and evidence:
                    evidence_card = [2] * len(evidence)
            
            while len(evidence_card) <= parent_idx:
                evidence_card.append(2) 
                
            parent_card = evidence_card[parent_idx]
            
            if values.ndim == 1:
                return 0.1  
            
            if values.ndim == 2:
                marginal_child = np.mean(values, axis=1)
            else:
                axes_to_sum = tuple(range(1, values.ndim))
                marginal_child = np.sum(values, axis=axes_to_sum)
            
            marginal_child = marginal_child / (np.sum(marginal_child) + 1e-12)
            marginal_entropy = -np.sum(marginal_child * np.log2(marginal_child + 1e-12))
            
            conditional_entropy = 0.0
            
            if values.ndim == 2 and len(evidence) == 1:
                parent_marginal = np.mean(values, axis=0)
                parent_marginal = parent_marginal / (np.sum(parent_marginal) + 1e-12)
                
                for p_val in range(min(parent_card, values.shape[1])):
                    if parent_marginal[p_val] > 1e-12:
                        conditional_dist = values[:, p_val]
                        conditional_dist = conditional_dist / (np.sum(conditional_dist) + 1e-12)
                        cond_ent = -np.sum(conditional_dist * np.log2(conditional_dist + 1e-12))
                        conditional_entropy += parent_marginal[p_val] * cond_ent
            else:
                total_configs = values.shape[1] if values.ndim == 2 else np.prod(values.shape[1:])
                configs_per_parent = max(1, total_configs // parent_card)
                
                for p_val in range(parent_card):
                    start_idx = p_val * configs_per_parent
                    end_idx = min((p_val + 1) * configs_per_parent, total_configs)
                    
                    if values.ndim == 2:
                        slice_values = values[:, start_idx:end_idx]
                        conditional_dist = np.mean(slice_values, axis=1) if slice_values.shape[1] > 0 else marginal_child
                    else:
                        flat_values = values.reshape(values.shape[0], -1)
                        slice_values = flat_values[:, start_idx:end_idx]
                        conditional_dist = np.mean(slice_values, axis=1) if slice_values.shape[1] > 0 else marginal_child
                    
                    conditional_dist = conditional_dist / (np.sum(conditional_dist) + 1e-12)
                    cond_ent = -np.sum(conditional_dist * np.log2(conditional_dist + 1e-12))
                    conditional_entropy += cond_ent / parent_card
            
            info_gain = marginal_entropy - conditional_entropy
            max_possible_gain = marginal_entropy
            normalized_gain = info_gain / (max_possible_gain + 1e-12)
            
            return max(0.0, min(1.0, normalized_gain))
            
        except Exception as e:
            if 'evidence_card' in str(e):
                print(f"Warning: Missing evidence_card for {parent}->{child}, using default cardinalities")
            else:
                print(f"Warning: Error measuring dependency strength for {parent}->{child}: {e}")
            
            evidence = child_cpt.get('evidence', [])
            return 1.0 / len(evidence) if parent in evidence else 0.0
    
    def cyclicity_check(self, parent, child, current_edges):
        """
        Check if adding parent->child would create a cycle
        """
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            G.add_edges_from(current_edges)
            G.add_edge(parent, child)
            
            return not nx.is_directed_acyclic_graph(G)
        
        except:
            def has_path(start, end, edges, visited=None):
                if visited is None:
                    visited = set()
                if start == end:
                    return True
                if start in visited:
                    return False
                visited.add(start)
                
                for p, c in edges:
                    if p == start and has_path(c, end, edges, visited.copy()):
                        return True
                return False
            
            return has_path(child, parent, current_edges)


    def estimate_edge_addition_cost(self, parent, child, current_cpts):
        """
        Estimate memory cost of adding an edge
        """
        child_cpt = next((c for c in current_cpts if c['variable'] == child), None)
        if not child_cpt:
            return 1000
        
        current_size = np.prod(child_cpt['cardinality'])
        parent_card = self.get_variable_cardinality(parent, current_cpts)
        new_size = current_size * parent_card
        
        return new_size - current_size


    def get_variable_cardinality(self, variable, cpts):
        """
        Get cardinality of a variable from CPTs - FIXED to use cardinality properly
        """
        for cpt in cpts:
            if cpt['variable'] == variable:
                return cpt['variable_card']
            elif variable in cpt.get('evidence', []):
                evidence = cpt['evidence']
                idx = evidence.index(variable)
                cardinality = cpt.get('cardinality', [])
                if len(cardinality) > idx + 1: 
                    return cardinality[idx + 1]
        
        return 2  

    def add_edge_to_cpts(self, parent, child, current_cpts):
        """
        Add an edge by updating the child's CPT to include the new parent
        """
        updated_cpts = []
        
        for cpt in current_cpts:
            if cpt['variable'] == child:
                new_evidence = list(cpt['evidence']) + [parent]
                updated_cpt = self.recompute_cpt_with_new_structure(cpt, new_evidence, current_cpts)
                updated_cpts.append(updated_cpt)
            else:
                updated_cpts.append(cpt)
        
        return updated_cpts


    def remove_edge_from_cpts(self, parent, child, current_cpts):
        """
        Remove an edge by updating the child's CPT to exclude the parent
        """
        updated_cpts = []
        
        for cpt in current_cpts:
            if cpt['variable'] == child and parent in cpt.get('evidence', []):
                new_evidence = [p for p in cpt['evidence'] if p != parent]
                updated_cpt = self.recompute_cpt_with_new_structure(cpt, new_evidence, current_cpts)
                updated_cpts.append(updated_cpt)
            else:
                updated_cpts.append(cpt)
        
        return updated_cpts


    def ensure_cpt_consistency(self, cpt_dict):
        """
        Ensure CPT has consistent cardinality and evidence_card fields
        """
        variable_card = cpt_dict.get('variable_card', 2)
        evidence = cpt_dict.get('evidence', [])
        
        cardinality = cpt_dict.get('cardinality', [])
        
        if not cardinality:
            evidence_card = cpt_dict.get('evidence_card', [2] * len(evidence))
            cardinality = [variable_card] + list(evidence_card)
            cpt_dict['cardinality'] = cardinality
        
        if len(cardinality) > 1:
            evidence_card = cardinality[1:]
        else:
            evidence_card = []
        
        while len(evidence_card) < len(evidence):
            evidence_card.append(2) 
            
        cpt_dict['evidence_card'] = evidence_card
        cpt_dict['cardinality'] = [variable_card] + list(evidence_card)
        
        return cpt_dict


    def recompute_cpt_with_new_structure(self, original_cpt, new_evidence, all_cpts):
        """
        Recompute CPT with new parent structure - FIXED cardinality handling
        """
        variable = original_cpt['variable']
        var_card = original_cpt['variable_card']
        
        new_evidence_cards = []
        for ev in new_evidence:
            ev_card = 2 
            for cpt in all_cpts:
                if cpt['variable'] == ev:
                    ev_card = cpt['variable_card']
                    break
                elif ev in cpt.get('evidence', []):
                    evidence = cpt['evidence']
                    idx = evidence.index(ev)
                    cardinality = cpt.get('cardinality', [])
                    if len(cardinality) > idx + 1:
                        ev_card = cardinality[idx + 1]
                    break
            new_evidence_cards.append(ev_card)
        
        if new_evidence_cards:
            required_configs = np.prod(new_evidence_cards)
            new_shape = [var_card] + new_evidence_cards
        else:
            required_configs = 1
            new_shape = [var_card, 1]
        
        new_values = np.ones((var_card, required_configs)) / var_card
        
        old_evidence = original_cpt['evidence']
        old_values = np.array(original_cpt['values'])
        
        if old_values.ndim > 2:
            old_values = old_values.reshape(var_card, -1)
        elif old_values.ndim == 1:
            old_values = old_values.reshape(var_card, 1)
        elif old_values.ndim == 0:
            old_values = np.array([[old_values]])
        
        common_evidence = [ev for ev in new_evidence if ev in old_evidence]
        if len(common_evidence) == len(old_evidence) == len(new_evidence):
            if old_values.shape[1] == required_configs:
                new_values = old_values
            else:
                marginal = np.mean(old_values, axis=1, keepdims=True)
                new_values = np.tile(marginal / np.sum(marginal), (1, required_configs))
        elif common_evidence:
            marginal = np.mean(old_values, axis=1, keepdims=True)
            marginal = marginal / np.sum(marginal)
            new_values = np.tile(marginal, (1, required_configs))

        new_values = new_values / np.sum(new_values, axis=0, keepdims=True)
        
        if len(new_evidence_cards) == 0:
            new_values = new_values.flatten()
        elif len(new_evidence_cards) == 1:
            new_values = new_values  
        else:
            if new_values.size == np.prod(new_shape):
                new_values = new_values.reshape(new_shape)
        
        new_cardinality = [var_card] + new_evidence_cards
        
        return {
            'variable': variable,
            'variable_card': var_card,
            'values': new_values,
            'evidence': new_evidence,
            'evidence_card': new_evidence_cards, 
            'cardinality': new_cardinality  
        }

    def build_model_from_cpts(self, cpts: List[Dict[str, Any]], forbidden_edges: List[Tuple[str, str]] = None) -> DiscreteBayesianNetwork:
        """
        Build a complete Bayesian network model from conditional probability tables.
        
        Args:
            cpts: List of CPT dictionaries containing variable information
            
        Returns:
            A complete DiscreteBayesianNetwork with structure and parameters
        """
        model = DiscreteBayesianNetwork()

        # forbidden_edges_set = set(forbidden_edges) if forbidden_edges else set()
        # if forbidden_edges_set:
        #     print(f"Forbidden edges: {forbidden_edges_set}")
        
        variable_cardinality: Dict[str, int] = {}
        for cpt in cpts:
            variable_cardinality[cpt['variable']] = cpt['variable_card']
            for ev, ev_card in zip(cpt['evidence'], cpt['evidence_card']):
                if ev not in variable_cardinality:
                    variable_cardinality[ev] = ev_card
        
        model.add_nodes_from(variable_cardinality.keys())
        
        edges_added = 0
        edges_skipped_forbidden = 0
        edges_skipped_cycle = 0

        for cpt in cpts:
            child = cpt['variable']
            for parent in cpt['evidence']:
                edge = (parent, child)
                # Check if edge is forbidden
                # if edge in forbidden_edges_set:
                #     edges_skipped_forbidden += 1
                #     print(f"Skipped forbidden edge: {parent} -> {child}")
                #     continue

                if not model.has_edge(parent, child):
                    try:
                        model.add_edge(parent, child)
                    except ValueError as e:
                        if "loop" in str(e).lower():
                            print(f"Skipped edge {parent} -> {child} to avoid cycle")
                        else:
                            raise e
                        
        print(f"Edge statistics: {edges_added} added, {edges_skipped_forbidden} forbidden, {edges_skipped_cycle} cycle-prevented")
        
        # if forbidden_edges_set:
        #     actual_forbidden = [edge for edge in model.edges() if edge in forbidden_edges_set]
        #     if actual_forbidden:
        #         print(f"[ERROR] Found forbidden edges in final model: {actual_forbidden}")
        #         for edge in actual_forbidden:
        #             model.remove_edge(*edge)
        #             print(f"Removed forbidden edge: {edge}")
        #     else:
        #         print(f"[SUCCESS] No forbidden edges found in final model")
        
        def marginalize_cpd_values(values_array: np.ndarray, 
                                 original_evidence: List[str], 
                                 original_evidence_card: List[int], 
                                 kept_parents: List[str]) -> np.ndarray:
            """
            Marginalize CPD values when some parents are dropped due to cycle prevention.
            
            Args:
                values_array: Original CPD values
                original_evidence: Original parent variables
                original_evidence_card: Original parent cardinalities
                kept_parents: Parents that remain in the final structure
                
            Returns:
                Marginalized CPD values
            """
            if not kept_parents:
                marginalized = values_array
                for _ in range(len(original_evidence)):
                    marginalized = marginalized.sum(axis=1)  
                return marginalized.reshape(-1, 1)
            
            if len(kept_parents) == len(original_evidence) and all(p in original_evidence for p in kept_parents):
                return values_array
        
            kept_indices = []
            for parent in kept_parents:
                if parent in original_evidence:
                    kept_indices.append(original_evidence.index(parent))
            
            if not kept_indices:
                marginalized = values_array
                for _ in range(len(original_evidence)):
                    marginalized = marginalized.sum(axis=1)
                return marginalized.reshape(-1, 1)
            
            all_indices = list(range(len(original_evidence)))
            drop_indices = [i for i in all_indices if i not in kept_indices]
            
            marginalized = values_array
            for drop_idx in sorted(drop_indices, reverse=True):
                marginalized = marginalized.sum(axis=drop_idx + 1)  
            
            if len(kept_indices) > 1:
                expected_order = [original_evidence.index(p) for p in kept_parents if p in original_evidence]
                current_order = sorted(kept_indices)
                if expected_order != current_order:
                    print(f"Warning: Parent order changed for some parents")
            
            return marginalized
        
        def normalize_cpd(values: np.ndarray) -> np.ndarray:
            """
            Normalize CPD values to ensure they sum to 1.0 along variable dimension.
            
            Args:
                values: CPD values to normalize
                
            Returns:
                Normalized CPD values
            """
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            elif values.ndim == 0:
                values = values.reshape(1, 1)
            
            col_sums = values.sum(axis=0, keepdims=True)
        
            col_sums[col_sums == 0] = 1e-12
            normalized = values / col_sums
            
            if normalized.ndim == 1:
                normalized = normalized.reshape(-1, 1)
            
            return normalized
        
        cpds: List[TabularCPD] = []
        for cpt in cpts:
            variable = cpt['variable']
            variable_card = variable_cardinality[variable]
            original_evidence = cpt['evidence']
            original_evidence_card = cpt['evidence_card']
            
            parents = list(model.predecessors(variable))  
            
            values_array = np.array(cpt['values'])
            
            marginalized_values = marginalize_cpd_values(
                values_array, original_evidence, original_evidence_card, parents
            )
            
            if marginalized_values.ndim == 1:
                marginalized_values = marginalized_values.reshape(-1, 1)
            elif marginalized_values.ndim == 0:
                marginalized_values = marginalized_values.reshape(1, 1)
            
            normalized_values = normalize_cpd(marginalized_values)
            
            evidence_card = [variable_cardinality[p] for p in parents] if parents else None
            
            if parents:
                expected_cols = np.prod(evidence_card)

                if normalized_values.ndim != 2:
                    if normalized_values.ndim == 1:
                        normalized_values = normalized_values.reshape(-1, 1)
                    else:
                        normalized_values = normalized_values.reshape(variable_card, -1)
                
                if normalized_values.shape[1] != expected_cols:

                    if normalized_values.size == variable_card * expected_cols:
                        normalized_values = normalized_values.reshape(variable_card, expected_cols)
                    else:
                        if normalized_values.shape[1] < expected_cols:

                            missing_factor = expected_cols // normalized_values.shape[1]
                            if expected_cols % normalized_values.shape[1] == 0:
                                normalized_values = np.repeat(normalized_values, missing_factor, axis=1)
                            else:
                                uniform_values = np.ones((variable_card, expected_cols)) / variable_card
                                normalized_values = uniform_values
                                print(f"Created uniform distribution for {variable}")
                        else:
                            normalized_values = normalized_values[:, :expected_cols]
                            print(f"Truncated columns for {variable}")
            else:
                if normalized_values.ndim != 2:
                    normalized_values = normalized_values.reshape(variable_card, 1)
            
            if normalized_values.ndim != 2:
                raise TypeError(f"After all processing, values for {variable} is still not 2D: shape={normalized_values.shape}, ndim={normalized_values.ndim}")
            
            cpd = TabularCPD(
                variable=variable,
                variable_card=variable_card,
                values=normalized_values,
                evidence=parents if parents else None,
                evidence_card=evidence_card
            )
            
            cpds.append(cpd)
        
        model.add_cpds(*cpds)
        
        if not model.check_model():
            print("Warning: Model validation failed.")
            for cpd in model.get_cpds():
                if not np.allclose(cpd.values.sum(axis=0), 1.0, rtol=1e-10):
                    print(f"CPD for {cpd.variable} does not sum to 1.0")

        # if forbidden_edges_set:
        #     final_forbidden = [edge for edge in model.edges() if edge in forbidden_edges_set]
        #     if final_forbidden:
        #         print(f"[CRITICAL ERROR] Forbidden edges found in final model: {final_forbidden}")
        #     else:
        #         print(f"[FINAL SUCCESS] Model built with {len(model.edges())} edges, no forbidden edges present")
        
        return model