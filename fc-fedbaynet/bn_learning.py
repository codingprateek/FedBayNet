import numpy as np
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
import json
import matplotlib.pyplot as plt
import networkx as nx
import traceback

class Client:

    def __init__(self):
        self.model = None
        self.dataset = None
        self.client_size = 0

    def convert_to_python_object(self, obj: Any) -> Any:
        """
        Converts objects into Python object to support network analysis.
        """
        if isinstance(obj, (list, tuple)):
            return [self.convert_to_python_object(i) for i in obj]
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        else:
            return obj
        
    def extract_cpts(self, model: Any = None) -> List[Dict[str, Any]]:
        """
        Extracts Conditional Probability Tables (CPTs) from the model.
        """
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("No model available.")
        
        if hasattr(model, 'edges') and not hasattr(model, 'get_cpts'):
            try:
                bn_model = DiscreteBayesianNetwork(model.edges())
                bn_model.fit(self.dataset, estimator=MaximumLikelihoodEstimator)
                model = bn_model
                print("Converted DAG to BayesianNetwork and fitted with data")
                
            except Exception as e:
                print(f"Warning: Could not convert DAG to BayesianNetwork: {str(e)}")
                return []
        
        elif hasattr(model, 'get_cpts') and len(model.get_cpds()) == 0:
            try:
                model.fit(self.dataset, estimator=MaximumLikelihoodEstimator)
            except Exception as e:
                print(f"Warning: Could not fit the model parameters: {str(e)}")
                return []
        
        cpts_list = []
        try:
            cpts = model.get_cpds()
            print(f"Number of CPTs: {len(cpts)}")
            
            for i, cpt in enumerate(cpts):
                print(f"Processing CPT {i}: {cpt.variable}")
                
                try:
                    evidence = cpt.get_evidence()
                    evidence_list = self.convert_to_python_object(evidence) if evidence else []
                except Exception as e:
                    print(f"Warning: Could not get evidence for {cpt.variable}: {str(e)}")
                    evidence_list = []
                
                try:
                    cardinality = self.convert_to_python_object(cpt.cardinality)
                except Exception as e:
                    print(f"Warning: Could not get cardinality for {cpt.variable}: {str(e)}")
                    cardinality = []
                
                try:
                    values = cpt.get_values()
                    values_converted = self.convert_to_python_object(values)
                except Exception as e:
                    print(f"Warning: Could not get values for {cpt.variable}: {str(e)}")
                    values_converted = []
                
                cpt_dict = {
                    "variable": cpt.variable,
                    "evidence": evidence_list,
                    "cardinality": cardinality,
                    "values": values_converted
                }
                cpts_list.append(cpt_dict)
                
        except Exception as e:
            print(f"Warning: Could not extract CPTs: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return cpts_list

    def save_cpts_to_json(self, filename: str = "cpts.json", model: Any = None) -> None:
        """
        Saves CPTs to JSON.
        """
        if model is None:
            model = self.model

        if model is None:
            print(f"No model available. Skipping CPT save to {filename}.")
            return 

        cpts_list = self.extract_cpts(model)

        if cpts_list:
            with open(filename, "w") as f:
                json.dump(cpts_list, f, indent=4)
            print(f"CPTs saved to: {filename}")
        else:
            print("Model has no CPTs to save.")

    def create_network_graph(self, model: Any = None) -> nx.Graph:
        """
        Returns the graph generated from a model.
        """
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("No model available.")

        graph = nx.Graph()
        if model is not None and len(model.edges()) > 0:
            graph.add_edges_from(model.edges())
        
        return graph
    
    def visualize_network(self, model: Any = None,
                      target_node: str = 'class',
                      target_color: str = '#9f0000',
                      node_color: str = '#22666F',
                      target_size: int = 1200,
                      node_size: int = 800,
                      seed: int = 23,
                      figsize: tuple = (10, 8),
                      save_path: Optional[str] = None,
                      title_prefix: str = "Bayesian Network") -> None:
        """
        Visualizes network structure.
        """
        if model is None:
            model = self.model
            
        if model is None:
            print("No model available to visualize.")
            return

        graph = self.create_network_graph(model)

        if len(graph.nodes()) == 0:
            print("No nodes available for visualization.")
            return

        _, ax = plt.subplots(figsize=figsize)
        
        pos = nx.spring_layout(graph, seed=seed)

        node_colors = [target_color if node == target_node else node_color
                    for node in graph.nodes()]
        node_sizes = [target_size if node == target_node else node_size
                    for node in graph.nodes()]

        nx.draw(
            graph,
            pos,
            with_labels=False,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color='gray',
            alpha=0.8,
            ax=ax 
        )

        nx.draw_networkx_labels(
            graph,
            pos,
            labels={node: node for node in graph.nodes()},
            font_color='black',
            font_weight='bold',
            ax=ax  
        )

        title = f"{title_prefix} - {len(graph.nodes())} nodes, {len(graph.edges())} edges"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to: {save_path}")

        plt.show()

    def learn_local_structure(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Learns Bayesian network structure from local data.
        """
        self.dataset = dataset
        self.client_size = dataset.shape[0]

        try:
            print("[CLIENT] Learning local Bayesian network structure...")
            hc = HillClimbSearch(self.dataset)
            structure = hc.estimate(scoring_method='bic-d')
            
            if structure is None:
                self.model = None
                return {"size": self.client_size, "cpts": []}

            bn_model = DiscreteBayesianNetwork(structure.edges())
            bn_model.fit(self.dataset, estimator=MaximumLikelihoodEstimator)
            self.model = bn_model
            
            print(f"[CLIENT] Learned network with {len(structure.edges())} edges")

            # CPTs from local model
            cpts_list = self.extract_cpts(self.model)

            return {"size": self.client_size, "cpts": cpts_list}

        except Exception as e:
            print(f"Error in learn_local_structure: {str(e)}")
            traceback.print_exc()
            self.model = None
            return {"size": self.client_size, "cpts": []}


class Coordinator(Client):
    def __init__(self):
        super().__init__()
        self.global_model = None
        
    def aggregate_cpts(self, clients_cpts: List[List[Dict[str, Any]]],
                       clients_weights: List[float]) -> List[Dict[str, Any]]:
        """Aggregates local cpts based on clients dataset size [Weighted Averaging]."""
        
        weights = np.array(clients_weights, dtype=float)
        weights = weights / weights.sum()

        # Collect all variables and their structures
        variable_map = {}
        all_variables = set()
        
        for client_idx, cpt_list in enumerate(clients_cpts):
            for cpt in cpt_list:
                var = cpt["variable"]
                all_variables.add(var)
                if var not in variable_map:
                    variable_map[var] = []
                variable_map[var].append((cpt, weights[client_idx]))

        print(f"[COORDINATOR] Aggregating {len(all_variables)} variables from {len(clients_cpts)} clients")

        aggregated_cpts = []

        for var in sorted(all_variables):
            if var not in variable_map:
                continue
                
            cpt_entries = variable_map[var]
            
            structures = {}
            for cpt_dict, weight in cpt_entries:
                evidence_key = tuple(sorted(cpt_dict["evidence"]))
                if evidence_key not in structures:
                    structures[evidence_key] = {"weight": 0, "example": cpt_dict}
                structures[evidence_key]["weight"] += weight
            
            # Use the structure with highest weight
            best_structure_key = max(structures.keys(), key=lambda k: structures[k]["weight"])
            best_structure = structures[best_structure_key]["example"]
            
            evidence = best_structure["evidence"]
            cardinality = best_structure["cardinality"]

            print(f"[COORDINATOR] Variable {var}: evidence={evidence}, participants={len(cpt_entries)}")

            # Aggregate values for CPTs with matching structure
            matching_cpts = [(cpt, w) for cpt, w in cpt_entries 
                           if tuple(sorted(cpt["evidence"])) == best_structure_key]

            # Get max shape across matching CPTs
            all_shapes = [np.array(cpt_dict["values"]).shape for cpt_dict, _ in matching_cpts]
            max_rows = max(s[0] for s in all_shapes)
            max_cols = max(s[1] if len(s) > 1 else 1 for s in all_shapes)

            aggregated_values = np.zeros((max_rows, max_cols))
            total_weight = 0

            for cpt_dict, w in matching_cpts:
                values = np.array(cpt_dict["values"], dtype=float)
                
                if values.ndim == 1:
                    values = values.reshape(-1, 1)

                padded = np.zeros((max_rows, max_cols))
                padded[:values.shape[0], :values.shape[1]] = values
                
                aggregated_values += w * padded
                total_weight += w

            if total_weight > 0:
                aggregated_values /= total_weight

            col_sums = aggregated_values.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1 
            aggregated_values /= col_sums

            if max_cols == 1:
                aggregated_values = aggregated_values.flatten()

            aggregated_cpts.append({
                "variable": var,
                "evidence": evidence,
                "cardinality": cardinality,
                "values": aggregated_values.tolist()
            })

        print(f"[COORDINATOR] Successfully aggregated {len(aggregated_cpts)} CPTs")
        return aggregated_cpts
    
    def create_global_network_from_cpts(self, aggregated_cpts: List[Dict[str, Any]]) -> Optional[DiscreteBayesianNetwork]:
        """Create a Bayesian network from aggregated CPTs"""
        try:
            # Extract structure from CPTs
            edges = []
            variables = set()
            
            for cpt_data in aggregated_cpts:
                var_name = cpt_data["variable"]
                variables.add(var_name)
                evidence = cpt_data.get('evidence', [])
                for parent in evidence:
                    edges.append((parent, var_name))
                    variables.add(parent)
            
            if not edges:
                print("[COORDINATOR] No edges found in aggregated CPTs")
                return None
            
            model = DiscreteBayesianNetwork(edges)
            
            cpts = []
            for cpt_data in aggregated_cpts:
                variable = cpt_data["variable"]
                evidence = cpt_data["evidence"]
                cardinality = cpt_data["cardinality"]
                values = np.array(cpt_data["values"])
                
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                elif values.ndim == 0:
                    values = values.reshape(1, 1)
                
                print(f"[COORDINATOR] Creating CPT for {variable}: values shape {values.shape}, cardinality {cardinality}")
                
                cpt = TabularCPD(
                    variable=variable,
                    variable_card=int(cardinality[0]) if isinstance(cardinality, list) else int(cardinality),
                    values=values,
                    evidence=evidence if evidence else None,
                    evidence_card=[int(c) for c in cardinality[1:]] if len(cardinality) > 1 else None
                )
                cpts.append(cpt)

            model.add_cpds(*cpts)
            self.global_model = model
            
            print(f"[COORDINATOR] Created global network with {len(edges)} edges and {len(cpts)} CPTs")
            return model
            
        except Exception as e:
            print(f"[COORDINATOR] Error creating global network: {str(e)}")
            traceback.print_exc()
            return None