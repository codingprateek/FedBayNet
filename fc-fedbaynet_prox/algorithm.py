import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from typing import Optional, Any, List, Dict, Tuple, Set
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BIC
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination
from sklearn.model_selection import KFold
import networkx as nx
import torch
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd


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
                         seed: int = 23,
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None,
                         title_prefix: str = "Bayesian Network") -> None:
        """
        Visualize a Bayesian network structure using NetworkX and Matplotlib.
        
        Args:
            structure: The Bayesian network structure to visualize
            target_node: Name of the target node to highlight
            target_color: Color for the target node
            node_color: Color for regular nodes
            target_size: Size of the target node
            node_size: Size of regular nodes
            seed: Random seed for layout positioning
            figsize: Figure size as (width, height)
            save_path: Optional path to save the visualization
            title_prefix: Prefix for the plot title
            
        Returns:
            None. Displays the network visualization.
        """
        graph = nx.Graph()
        if structure is not None and len(structure.edges()) > 0:
            graph.add_edges_from(structure.edges())
        
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


    def create_client_model(self, dataset: pd.DataFrame) -> DiscreteBayesianNetwork:
        """
        Create a Bayesian network model from dataset using Hill Climbing search.
        
        Args:
            dataset: Input dataset as a pandas DataFrame
            
        Returns:
            A fitted DiscreteBayesianNetwork model
        """
        hc = HillClimbSearch(dataset)
        structure = hc.estimate(scoring_method='bic-d')
        model = structure.fit(dataset, estimator=MaximumLikelihoodEstimator)
        return model


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
        model = model.fit(data, estimator=MaximumLikelihoodEstimator)

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


    def fedprox_local_update(self, client_cpts: List[Dict[str, Any]],
                     global_cpts: List[Dict[str, Any]],
                     data: pd.DataFrame,
                     mu: float = 1.0,  
                     lr: float = 0.1, 
                     epochs: int = 50) -> List[Dict[str, Any]]:  
        """
        Perform FedProx local CPT update given client data and global CPTs.
        
        If global_cpts is empty, the function performs standard MLE updates (mu=0).

        Args:
            client_cpts: List of dicts in CPT format (initial client CPTs)
            global_cpts: List of dicts in the CPT format (global CPTs)
            data: Client's local data
            mu: FedProx proximal coefficient
            lr: Learning rate
            epochs: Number of gradient steps
            
        Returns:
            Updated CPTs in the same dictionary format
        """
        updated_cpts = []

        use_prox = bool(global_cpts) and mu > 0.0

        for cpt_idx, cpt in enumerate(client_cpts):
            variable = cpt["variable"]
            parents = cpt["evidence"]
            var_card = cpt["variable_card"]

            if parents:
                parent_data = data[parents].values
                parent_cards = cpt["evidence_card"]
                config_idx = np.zeros(len(parent_data), dtype=int)
                
                for i, row in enumerate(parent_data):
                    config_val = 0
                    multiplier = 1
                    for j in range(len(parents) - 1, -1, -1):  
                        parent_val = int(row[j])
                        if parent_val > 0:
                            parent_val = parent_val - 1 if parent_val > 1 else parent_val
                        config_val += parent_val * multiplier
                        multiplier *= parent_cards[j]
                    config_idx[i] = config_val
            else:
                config_idx = np.zeros(len(data), dtype=int)

            if parents:
                num_configs = int(np.prod(cpt["evidence_card"]))
            else:
                num_configs = 1

            values_array = np.array(cpt["values"])
            
            if values_array.ndim > 2:
                values_array = values_array.reshape(var_card, -1)
            elif values_array.ndim == 1:
                values_array = values_array.reshape(var_card, -1)
            
            current_configs = values_array.shape[1]
            if current_configs != num_configs:
                if current_configs == 1 and num_configs > 1:
                    values_array = np.repeat(values_array, num_configs, axis=1)
                else:
                    if current_configs > num_configs:
                        values_array = values_array[:, :num_configs]
                    else:
                        padding_shape = (var_card, num_configs - current_configs)
                        padding = np.ones(padding_shape) / var_card
                        values_array = np.hstack([values_array, padding])
            
            values_array = np.clip(values_array, 1e-10, 1.0) 
            logits = torch.log(torch.tensor(values_array, dtype=torch.float32))

            if use_prox:
                global_cpt = global_cpts[cpt_idx]
                global_values = np.array(global_cpt["values"])
                global_var_card = global_cpt["variable_card"]
                
                if global_values.ndim > 2:
                    global_values = global_values.reshape(global_var_card, -1)
                elif global_values.ndim == 1:
                    global_values = global_values.reshape(global_var_card, -1)
                
                if global_var_card != var_card:
                    if global_var_card < var_card:
                        padding_rows = np.ones((var_card - global_var_card, global_values.shape[1])) / var_card
                        global_values = np.vstack([global_values, padding_rows])
                    else:
                        global_values = global_values[:var_card, :]
                        
                current_configs = global_values.shape[1]
                if current_configs != num_configs:
                    if current_configs == 1 and num_configs > 1:
                        global_values = np.repeat(global_values, num_configs, axis=1)
                    else:
                        if current_configs > num_configs:
                            global_values = global_values[:, :num_configs]
                        else:
                            padding_shape = (var_card, num_configs - current_configs) 
                            padding = np.ones(padding_shape) / var_card
                            global_values = np.hstack([global_values, padding])
                
                global_values = np.clip(global_values, 1e-10, 1.0)
                global_logits = torch.log(torch.tensor(global_values, dtype=torch.float32))
            else:
                global_logits = torch.zeros_like(logits)
                
            mu_local = mu if use_prox else 0.0

            logits = logits.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([logits], lr=lr)

            target_values = data[variable].values
            
            target_values = np.array([int(v) - 1 if int(v) > 0 and int(v) <= var_card else int(v) for v in target_values])
            target_values = np.clip(target_values, 0, var_card - 1)  
            target = F.one_hot(torch.tensor(target_values), num_classes=var_card).float()

            config_idx = np.clip(config_idx, 0, num_configs - 1)

            for _ in range(epochs):
                optimizer.zero_grad()
                row_logits = logits[:, config_idx].T 
                probs = F.softmax(row_logits, dim=1)
                nll = -torch.sum(target * torch.log(probs + 1e-10)) / len(data)
                
                prox = (mu_local / 2) * torch.sum((logits - global_logits)**2)
                loss = nll + prox
                loss.backward()
                optimizer.step()

            updated_values = F.softmax(logits.detach(), dim=0).T.tolist() 
            
            if len(updated_values) == 1:
                updated_values = updated_values[0] 
            else:
                updated_values = np.array(updated_values).T.tolist()  

            updated_cpt = {
                "variable": variable,
                "variable_card": var_card,
                "values": updated_values,
                "evidence": parents,
                "evidence_card": cpt["evidence_card"],
                "cardinality": cpt["cardinality"]
            }

            updated_cpts.append(updated_cpt)

        return updated_cpts


    def kl_divergence(self, p, q, epsilon=1e-12):
        p = np.asarray(p) + epsilon
        q = np.asarray(q) + epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))


    def cpd_similarity(self, cpd1, cpd2):
        try:
            values1 = cpd1.get_values().flatten()
            values2 = cpd2.get_values().flatten()
            if len(values1) != len(values2):
                return 0.0
            kl_12 = self.kl_divergence(values1, values2)
            kl_21 = self.kl_divergence(values2, values1)
            symmetric_kl = (kl_12 + kl_21) / 2
            return np.exp(-symmetric_kl)
        except:
            return 0.0


    def fuse_cpds(self, expert_cpd, data_cpd, expert_weight=0.8):
        try:
            expert_values = expert_cpd.get_values()
            data_values = data_cpd.get_values()
            if expert_values.shape != data_values.shape:
                return expert_cpd
            fused_values = expert_weight * expert_values + (1 - expert_weight) * data_values
            if fused_values.ndim == 1:
                fused_values = fused_values / np.sum(fused_values)
            else:
                fused_values = fused_values / np.sum(fused_values, axis=-1, keepdims=True)
            fused_cpd = TabularCPD(
                variable=expert_cpd.variable,
                variable_card=expert_cpd.variable_card,
                values=fused_values,
                evidence=expert_cpd.variables[1:] if len(expert_cpd.variables) > 1 else None,
                evidence_card=expert_cpd.cardinality[1:] if len(expert_cpd.cardinality) > 1 else None
            )
            return fused_cpd
        except:
            return expert_cpd


    def score_edge_operation_bic(self, fused_bn, data_bn, data_df, operation, edge):
        """Score an edge operation using CPD similarity + BIC improvement"""
        u, v = edge
        score = 0.0
        try:
            if operation in ["add", "reverse"]:
                cpd = data_bn.get_cpds(v if operation=="add" else u)
                uniform = np.ones_like(cpd.get_values().flatten()) / cpd.get_values().size
                score += self.kl_divergence(cpd.get_values().flatten(), uniform)
            
            temp_bn = fused_bn.copy()
            if operation == "add":
                temp_bn.add_edge(u, v)
            elif operation == "remove":
                temp_bn.remove_edge(u, v)
            elif operation == "reverse":
                temp_bn.remove_edge(u, v)
                temp_bn.add_edge(v, u)
            
            if nx.is_directed_acyclic_graph(temp_bn):
                bic = BIC(data_df)
                score += bic.score(temp_bn)
            else:
                score = -np.inf  
        except:
            score = -np.inf
        return score


    def fuse_bayesian_networks(self, expert_bn, data_bn, data_df,
                                        max_operations=5,
                                        similarity_threshold=0.3,
                                        expert_weight=0.8):
        """FedGES-style fusion with BIC-based scoring"""
        
        fused_bn = expert_bn.copy()
        common_nodes = set(expert_bn.nodes()).intersection(set(data_bn.nodes()))
        expert_edges = set(expert_bn.edges())
        data_edges = set(data_bn.edges())
        
        candidate_operations = []

        for u, v in data_edges - expert_edges:
            if u in common_nodes and v in common_nodes:
                temp_graph = nx.DiGraph(fused_bn.edges())
                temp_graph.add_edge(u, v)
                if nx.is_directed_acyclic_graph(temp_graph):
                    score = self.score_edge_operation_bic(fused_bn, data_bn, data_df, "add", (u, v))
                    candidate_operations.append(("add", (u, v), score))

        for u, v in expert_edges.intersection(data_edges):
            temp_graph = nx.DiGraph(fused_bn.edges())
            temp_graph.remove_edge(u, v)
            temp_graph.add_edge(v, u)
            if nx.is_directed_acyclic_graph(temp_graph):
                score = self.score_edge_operation_bic(fused_bn, data_bn, data_df, "reverse", (u, v))
                candidate_operations.append(("reverse", (u, v), score))

        for u, v in expert_edges - data_edges:
            temp_graph = nx.DiGraph(fused_bn.edges())
            temp_graph.remove_edge(u, v)
            if nx.is_directed_acyclic_graph(temp_graph):
                score = self.score_edge_operation_bic(fused_bn, data_bn, data_df, "remove", (u, v))
                candidate_operations.append(("remove", (u, v), score))

        candidate_operations.sort(key=lambda x: x[2], reverse=True)

        operations_done = 0
        for op, edge, score in candidate_operations:
            if operations_done >= max_operations:
                break
            u, v = edge
            try:
                if op == "add":
                    fused_bn.add_edge(u, v)
                    fused_bn.add_cpds(data_bn.get_cpds(v))
                elif op == "remove":
                    fused_bn.remove_edge(u, v)
                elif op == "reverse":
                    fused_bn.remove_edge(u, v)
                    fused_bn.add_edge(v, u)
                operations_done += 1
            except:
                continue

        for node in common_nodes:
            try:
                expert_cpd = expert_bn.get_cpds(node)
                data_cpd = data_bn.get_cpds(node)
                similarity = self.cpd_similarity(expert_cpd, data_cpd)
                if similarity < similarity_threshold:
                    fused_cpd = self.fuse_cpds(expert_cpd, data_cpd, expert_weight)
                    fused_bn.add_cpds(fused_cpd)
            except:
                continue

        return fused_bn


    def kfold_cv(self, data, model, target='class', k=5, csv_filename=None):
        """
        Perform K-Fold cross-validation on a Bayesian Network and return predictions with confidence.
        
        Parameters:
        - data: pandas DataFrame containing dataset
        - model: pgmpy DiscreteBaysianNetwork object (network structure)
        - target: name of target variable (default 'class')
        - k: number of folds (default 5)
        - csv_filename: name of CSV to save results (if None, don't save CSV)
        
        Returns:
        - avg_accuracy: average accuracy across folds
        """
        
        data = data.copy()  
        data[target] = data[target].astype(int)
        
        data['predicted_class'] = np.nan
        data['prediction_confidence'] = np.nan
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_accuracies = []

        for fold, (train_index, test_index) in enumerate(kf.split(data), 1):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            model.fit(train_data, estimator=MaximumLikelihoodEstimator)
            infer = VariableElimination(model)
            cpds = {node: model.get_cpds(node) for node in model.nodes()}

            correct = 0
            total = len(test_index)

            for i in test_index:
                evidence = {}
                for node in model.nodes():
                    if node != target:
                        val = int(data.iloc[i][node])
                        if val in cpds[node].state_names[node]:
                            evidence[node] = val

                try:
                    class_dist = infer.query(variables=[target], evidence=evidence, show_progress=False)
                    pred_idx = np.argmax(class_dist.values)
                    pred = int(cpds[target].state_names[target][pred_idx])
                    confidence = round(float(class_dist.values[pred_idx]), 4)
                except:
                    pred = int(train_data[target].mode()[0])
                    confidence = 1.0

                data.at[i, 'predicted_class'] = pred
                data.at[i, 'prediction_confidence'] = confidence

                if pred == data.iloc[i][target]:
                    correct += 1

            fold_acc = correct / total
            fold_accuracies.append(fold_acc)
            print(f"Fold {fold} Accuracy: {fold_acc:.4f}")

        avg_accuracy = np.mean(fold_accuracies)
        print(f"Average Accuracy across all folds: {avg_accuracy:.4f}")

        data['predicted_class'] = data['predicted_class'].astype(int)
        data['prediction_confidence'] = data['prediction_confidence'].astype(float)

        # Only save CSV if filename is provided
        if csv_filename is not None:
            data.to_csv(csv_filename, index=False)
            print(f"CSV saved as '{csv_filename}' with predicted class and confidence.")

        return round(float(avg_accuracy), 4)


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


    def aggregate_cpts(self, cpt_lists, weights):
        """
        Original CPT aggregation function (preserved as-is)
        """
        if len(cpt_lists) != len(weights):
            raise ValueError("Number of CPT lists must match number of weights")
        
        for i, cpt_list in enumerate(cpt_lists):
            if not isinstance(cpt_list, list):
                raise TypeError(f"CPT list at index {i} is not a list: {type(cpt_list)}")
            for j, cpt in enumerate(cpt_list):
                if not isinstance(cpt, dict):
                    raise TypeError(f"CPT at index {j} in list {i} is not a dict: {type(cpt)}")
                required_keys = ['variable', 'variable_card', 'values', 'evidence', 'evidence_card', 'cardinality']
                for key in required_keys:
                    if key not in cpt:
                        raise KeyError(f"CPT at index {j} in list {i} missing key: {key}")

        all_variables = set()
        for cpt_list in cpt_lists:
            for cpt in cpt_list:
                all_variables.add(cpt['variable'])
        all_variables = sorted(list(all_variables)) 
        
        aggregated_cpts = []
        
        for var in all_variables:
            var_cpts = []
            cpt_weights = []
            for i, cpt_list in enumerate(cpt_lists):
                cpt = next((c for c in cpt_list if c['variable'] == var), None)
                if cpt is not None:
                    var_cpts.append(cpt)
                    cpt_weights.append(weights[i])
            
            if not var_cpts:
                continue  
            
            evidence_sets = [set(cpt['evidence']) for cpt in var_cpts]
            all_evidence = sorted(list(set.union(*evidence_sets)))  
            
            max_evidence_cards = []
            for ev in all_evidence:
                max_card = 1
                for cpt in var_cpts:
                    if ev in cpt['evidence']:
                        idx = cpt['evidence'].index(ev)
                        max_card = max(max_card, cpt['evidence_card'][idx])
                max_evidence_cards.append(max_card)
            
            padded_values = []
            for cpt, weight in zip(var_cpts, cpt_weights):
                curr_shape = [cpt['variable_card']] + list(cpt['evidence_card'])
                curr_evidence = cpt['evidence']
                target_shape = [cpt['variable_card']] + max_evidence_cards
                
                values = cpt['values']
                if not isinstance(values, np.ndarray):
                    values = np.array(values)
                
                if values.shape != tuple(curr_shape):
                    values = np.reshape(values, curr_shape)
                
                if curr_evidence:
                    expanded_values = values
                    
                    curr_to_target_map = []
                    for i, ev in enumerate(all_evidence):
                        if ev in curr_evidence:
                            curr_idx = curr_evidence.index(ev)
                            curr_to_target_map.append(curr_idx + 1) 
                        else:
                            curr_to_target_map.append(-1) 
                    
                    new_axes_order = [0]
                    singleton_positions = []
                    
                    for i, (ev, target_pos) in enumerate(zip(all_evidence, curr_to_target_map)):
                        if target_pos != -1:
                            new_axes_order.append(target_pos)
                        else:
                            singleton_positions.append(i + 1)
                    
                    if len(new_axes_order) > 1:
                        expanded_values = np.transpose(expanded_values, new_axes_order)
                    
                    for pos in singleton_positions:
                        expanded_values = np.expand_dims(expanded_values, axis=pos)
                    
                    final_values = np.zeros(target_shape, dtype=float)
                    slices = [slice(0, cpt['variable_card'])]  
                    for i, (ev, max_card) in enumerate(zip(all_evidence, max_evidence_cards)):
                        if ev in curr_evidence:
                            curr_idx = curr_evidence.index(ev)
                            curr_card = cpt['evidence_card'][curr_idx]
                            slices.append(slice(0, curr_card))
                        else:
                            slices.append(slice(0, 1))
                    
                    final_values[tuple(slices)] = expanded_values
                    padded_values.append(final_values * weight)
                    
                else:
                    expanded_values = values
                    for _ in max_evidence_cards:
                        expanded_values = expanded_values[..., np.newaxis]
                    
                    final_values = np.broadcast_to(expanded_values, target_shape)
                    padded_values.append(final_values * weight)
        
            max_var_card = max(cpt['variable_card'] for cpt in var_cpts)
            final_shape = [max_var_card] + max_evidence_cards
            
            final_padded_values = []
            
            for padded_val, weight in zip(padded_values, cpt_weights):
                if padded_val.shape[0] < max_var_card:
                    final_padded = np.zeros(final_shape, dtype=float)
                    slices = [slice(0, padded_val.shape[0])] + [slice(0, s) for s in padded_val.shape[1:]]
                    final_padded[tuple(slices)] = padded_val
                    final_padded_values.append(final_padded)
                else:
                    final_padded_values.append(padded_val)
            
            total_weight = sum(cpt_weights)
            if total_weight > 0:
                aggregated_values = np.sum([val * weight for val, weight in zip(final_padded_values, cpt_weights)], axis=0) / total_weight
            else:
                aggregated_values = np.sum(final_padded_values, axis=0)
            
            values_reshaped = aggregated_values.reshape(max_var_card, -1)
            
            column_sums = values_reshaped.sum(axis=0)
            zero_columns = column_sums == 0
            values_normalized = values_reshaped.copy()
            
            if np.any(~zero_columns):
                values_normalized[:, ~zero_columns] = values_reshaped[:, ~zero_columns] / column_sums[~zero_columns]
            
            if np.any(zero_columns):
                values_normalized[:, zero_columns] = 1.0 / max_var_card
            
            final_aggregated_values = values_normalized.reshape(final_shape)
            
            new_cpt = {
                'variable': var,
                'variable_card': max_var_card,
                'values': final_aggregated_values,
                'evidence': all_evidence,
                'evidence_card': np.array(max_evidence_cards, dtype=int),
                'cardinality': np.array([max_var_card] + max_evidence_cards, dtype=int)
            }
            aggregated_cpts.append(new_cpt)
        
        return aggregated_cpts


    def calculate_mutual_information(self, cpt, var1_idx, var2_idx, normalize=True):
        """
        Calculate mutual information between two variables in a CPT
        
        Args:
            cpt: CPT dictionary
            var1_idx: Index of first variable (0 for target variable, >0 for evidence variables)
            var2_idx: Index of second variable
            normalize: Whether to normalize by joint entropy
        
        Returns:
            Mutual information value
        """
        values = cpt['values']
        
        if var1_idx == 0:  
            axes_to_sum = list(range(1, len(values.shape)))
            if var2_idx != 0:
                axes_to_sum.remove(var2_idx)
            p1 = np.sum(values, axis=tuple(axes_to_sum))
        else:  
            axes_to_sum = [0] + [i for i in range(1, len(values.shape)) if i != var1_idx]
            if var2_idx in axes_to_sum:
                axes_to_sum.remove(var2_idx)
            p1 = np.sum(values, axis=tuple(axes_to_sum))
        
        if var2_idx == 0:  
            axes_to_sum = list(range(1, len(values.shape)))
            if var1_idx != 0 and var1_idx in axes_to_sum:
                axes_to_sum.remove(var1_idx)
            p2 = np.sum(values, axis=tuple(axes_to_sum))
        else:  
            axes_to_sum = [0] + [i for i in range(1, len(values.shape)) if i != var2_idx]
            if var1_idx in axes_to_sum:
                axes_to_sum.remove(var1_idx)
            p2 = np.sum(values, axis=tuple(axes_to_sum))
        
        axes_to_sum = [i for i in range(len(values.shape)) if i not in [var1_idx, var2_idx]]
        if axes_to_sum:
            p_joint = np.sum(values, axis=tuple(axes_to_sum))
        else:
            p_joint = values
        
        # Calculate MI
        mi = 0.0
        for i in range(p1.shape[0] if len(p1.shape) > 0 else 1):
            for j in range(p2.shape[0] if len(p2.shape) > 0 else 1):
                if len(p1.shape) == 0:
                    p1_val = p1
                else:
                    p1_val = p1[i] if i < len(p1) else 0
                    
                if len(p2.shape) == 0:
                    p2_val = p2
                else:
                    p2_val = p2[j] if j < len(p2) else 0
                
                if len(p_joint.shape) == 0:
                    p_joint_val = p_joint
                elif len(p_joint.shape) == 1:
                    p_joint_val = p_joint[i] if i < len(p_joint) else 0
                else:
                    p_joint_val = p_joint[i, j] if i < p_joint.shape[0] and j < p_joint.shape[1] else 0
                
                if p_joint_val > 0 and p1_val > 0 and p2_val > 0:
                    mi += p_joint_val * np.log2(p_joint_val / (p1_val * p2_val))
        
        if normalize and mi > 0:
            h_joint = -np.sum(p_joint * np.log2(p_joint + 1e-12))
            if h_joint > 0:
                mi = mi / h_joint
        
        return max(0, mi) 


    def calculate_conditional_independence_score(self, cpt, var1, var2, conditioning_vars):
        """
        Calculate score for conditional independence test
        Lower score indicates stronger conditional independence
        """
        try:
            var1_idx = 0 if cpt['variable'] == var1 else (cpt['evidence'].index(var1) + 1 if var1 in cpt['evidence'] else None)
            var2_idx = 0 if cpt['variable'] == var2 else (cpt['evidence'].index(var2) + 1 if var2 in cpt['evidence'] else None)
            
            if var1_idx is None or var2_idx is None:
                return float('inf') 
            
            mi = self.calculate_mutual_information(cpt, var1_idx, var2_idx, normalize=True)
            
            # Penalize based on number of conditioning variables
            complexity_penalty = 0.01 * len(conditioning_vars)
            
            return mi + complexity_penalty
            
        except Exception:
            return float('inf')


    def learn_network_structure(self, cpt_lists, weights, edge_threshold=0.05, remove_threshold=0.01):
        """
        Learn network structure by aggregating CPTs and applying structure learning
        
        Args:
            cpt_lists: List of CPT lists to aggregate
            weights: Weights for aggregation
            edge_threshold: Threshold for adding new edges (higher MI = add edge)
            remove_threshold: Threshold for removing edges (lower CI score = remove edge)
        
        Returns:
            Updated CPTs with modified structure
        """
        aggregated_cpts = self.aggregate_cpts(cpt_lists, weights)
        
        variables = [cpt['variable'] for cpt in aggregated_cpts]
        all_variables = set(variables)
        
        current_edges = set()
        for cpt in aggregated_cpts:
            for parent in cpt['evidence']:
                current_edges.add((parent, cpt['variable']))
        
        proposed_edges = set()
        edges_to_remove = set()
        
        for var1 in all_variables:
            for var2 in all_variables:
                if var1 != var2 and (var1, var2) not in current_edges:
                  
                    mi_score = self.calculate_pairwise_mi(aggregated_cpts, var1, var2)
                    
                    if mi_score > edge_threshold:
                        proposed_edges.add((var1, var2))
                        
                        var2_cpt = next((c for c in aggregated_cpts if c['variable'] == var2), None)
                        if var2_cpt:
                            for existing_parent in var2_cpt['evidence']:
                                ci_score = self.calculate_conditional_mi(
                                    aggregated_cpts, existing_parent, var2, [var1]
                                )
                                
                                if ci_score < remove_threshold:
                                    edges_to_remove.add((existing_parent, var2))
        
        modified_cpts = []
        
        for cpt in aggregated_cpts:
            new_evidence = list(cpt['evidence'])
            child = cpt['variable']
            
            for parent, c in proposed_edges:
                if c == child and parent not in new_evidence:
                    new_evidence.append(parent)
            
            for parent, c in edges_to_remove:
                if c == child and parent in new_evidence:
                    new_evidence.remove(parent)
            
            if set(new_evidence) != set(cpt['evidence']):
                modified_cpt = self.recompute_cpt_with_new_structure(
                    cpt, new_evidence, aggregated_cpts
                )
                modified_cpts.append(modified_cpt)
            else:
                modified_cpts.append(cpt)
        
        return modified_cpts


    def calculate_pairwise_mi(self, cpts, var1, var2):
        """
        Calculate mutual information between two variables using pairwise approach
        This avoids creating the full joint distribution
        """
        relevant_cpts = []
        
        for cpt in cpts:
            cpt_vars = {cpt['variable']} | set(cpt['evidence'])
            if var1 in cpt_vars and var2 in cpt_vars:
                relevant_cpts.append(cpt)
        
        if not relevant_cpts:
            return self.calculate_mi_through_chain(cpts, var1, var2)
        
        best_cpt = min(relevant_cpts, key=lambda x: np.prod(x['cardinality']))
        
        var1_idx = None
        var2_idx = None
        
        if best_cpt['variable'] == var1:
            var1_idx = 0
        elif var1 in best_cpt['evidence']:
            var1_idx = best_cpt['evidence'].index(var1) + 1
            
        if best_cpt['variable'] == var2:
            var2_idx = 0
        elif var2 in best_cpt['evidence']:
            var2_idx = best_cpt['evidence'].index(var2) + 1
        
        if var1_idx is None or var2_idx is None:
            return 0.0
        
        values = best_cpt['values']
        
        axes_to_sum = [i for i in range(len(values.shape)) if i != var1_idx]
        p1 = np.sum(values, axis=tuple(axes_to_sum))
        
        axes_to_sum = [i for i in range(len(values.shape)) if i != var2_idx]
        p2 = np.sum(values, axis=tuple(axes_to_sum))
        
        axes_to_sum = [i for i in range(len(values.shape)) if i not in [var1_idx, var2_idx]]
        if axes_to_sum:
            p12 = np.sum(values, axis=tuple(axes_to_sum))
        else:
            p12 = values
        
        mi = 0.0
        p1 = p1 / np.sum(p1)  
        p2 = p2 / np.sum(p2) 
        p12 = p12 / np.sum(p12) 
        
        p1_flat = p1.flatten()
        p2_flat = p2.flatten()
        p12_flat = p12.flatten()
        
        if p12.ndim == 1:
            expected_size = len(p1_flat) * len(p2_flat)
            if len(p12_flat) == expected_size:
                p12 = p12_flat.reshape(len(p1_flat), len(p2_flat))
            else:
                p12 = np.outer(p1_flat, p2_flat)
        elif p12.ndim > 2:
            if p12.size == len(p1_flat) * len(p2_flat):
                p12 = p12_flat.reshape(len(p1_flat), len(p2_flat))
            else:
                while p12.ndim > 2:
                    p12 = np.sum(p12, axis=-1)
                if p12.shape != (len(p1_flat), len(p2_flat)):
                    p12 = np.outer(p1_flat, p2_flat)
        
        for i in range(len(p1_flat)):
            for j in range(len(p2_flat)):
                if i < p12.shape[0] and j < p12.shape[1]:
                    p12_val = p12[i, j]
                    p1_val = p1_flat[i]
                    p2_val = p2_flat[j]
                    
                    if p12_val > 1e-12 and p1_val > 1e-12 and p2_val > 1e-12:
                        mi += p12_val * np.log2(p12_val / (p1_val * p2_val))
        
        return max(0, mi)


    def calculate_mi_through_chain(self, cpts, var1, var2):
        """
        Calculate MI between two variables that don't appear in the same CPT
        by finding a connecting path through the network
        """
        graph = defaultdict(set)
        for cpt in cpts:
            child = cpt['variable']
            for parent in cpt['evidence']:
                graph[parent].add(child)
                graph[child].add(parent) 
        
        # Find shortest path between var1 and var2
        def find_path(start, end, visited=None):
            if visited is None:
                visited = set()
            
            if start == end:
                return [start]
            
            if start in visited:
                return None
            
            visited.add(start)
            
            for neighbor in graph[start]:
                path = find_path(neighbor, end, visited.copy())
                if path:
                    return [start] + path
            
            return None
        
        path = find_path(var1, var2)
        
        if path is None or len(path) < 2:
            return 0.0
        
        # Calculate MI along the path 
        # For a path A->B->C, MI(A,C) is approximated by the minimum MI along the path
        min_mi = float('inf')
        
        for i in range(len(path) - 1):
            v1, v2 = path[i], path[i + 1]
            
            # Find CPT containing both variables
            for cpt in cpts:
                cpt_vars = {cpt['variable']} | set(cpt['evidence'])
                if v1 in cpt_vars and v2 in cpt_vars:
                    mi = self.calculate_pairwise_mi(cpts, v1, v2)
                    min_mi = min(min_mi, mi)
                    break
        
        decay_factor = 0.8 ** (len(path) - 2) 
        
        return max(0, min_mi * decay_factor if min_mi != float('inf') else 0.0)


    def calculate_conditional_mi(self, cpts, var1, var2, conditioning_vars):
        """
        Calculate conditional MI without constructing full joint distribution
        """
        required_vars = {var1, var2} | set(conditioning_vars)
        
        relevant_cpts = []
        for cpt in cpts:
            cpt_vars = {cpt['variable']} | set(cpt['evidence'])
            if required_vars.issubset(cpt_vars):
                relevant_cpts.append(cpt)
        
        if not relevant_cpts:
            # Can't compute exact conditional MI, return approximation
            # If conditioning variables make var1 and var2 independent, 
            # their unconditional MI should be low
            uncond_mi = self.calculate_pairwise_mi(cpts, var1, var2)
            
            # If we have strong connections to conditioning vars, the conditional MI should be lower
            var1_to_cond = max([self.calculate_pairwise_mi(cpts, var1, cond) for cond in conditioning_vars] + [0])
            var2_to_cond = max([self.calculate_pairwise_mi(cpts, var2, cond) for cond in conditioning_vars] + [0])
            
            # If both variables are strongly connected to conditioning variables,
            # reduce the conditional MI
            reduction_factor = min(1.0, (var1_to_cond + var2_to_cond) / 2.0)
            
            return uncond_mi * (1 - reduction_factor)
        
        best_cpt = min(relevant_cpts, key=lambda x: np.prod(x['cardinality']))
        
        var1_idx = None
        var2_idx = None
        cond_indices = []
        
        if best_cpt['variable'] == var1:
            var1_idx = 0
        elif var1 in best_cpt['evidence']:
            var1_idx = best_cpt['evidence'].index(var1) + 1
            
        if best_cpt['variable'] == var2:
            var2_idx = 0
        elif var2 in best_cpt['evidence']:
            var2_idx = best_cpt['evidence'].index(var2) + 1
        
        for cond_var in conditioning_vars:
            if best_cpt['variable'] == cond_var:
                cond_indices.append(0)
            elif cond_var in best_cpt['evidence']:
                cond_indices.append(best_cpt['evidence'].index(cond_var) + 1)
        
        if var1_idx is None or var2_idx is None or len(cond_indices) != len(conditioning_vars):
            return 0.0
        
        values = best_cpt['values']
        values = values / np.sum(values)  
        
        all_relevant_axes = [var1_idx, var2_idx] + cond_indices
        
        # Marginalize to get P(var1, var2, cond_vars)
        axes_to_sum = [i for i in range(len(values.shape)) if i not in all_relevant_axes]
        if axes_to_sum:
            relevant_dist = np.sum(values, axis=tuple(axes_to_sum))
        else:
            relevant_dist = values
        
        # If the conditional distribution is uniform,
        # then conditional MI is low
        if relevant_dist.size == 0:
            return 0.0
        
        # More uniform = lower conditional dependence
        entropy = -np.sum(relevant_dist * np.log2(relevant_dist + 1e-12))
        max_entropy = np.log2(relevant_dist.size)
        
        if max_entropy > 0:
            uniformity = entropy / max_entropy
            # High uniformity suggests low conditional dependence
            return (1 - uniformity) * self.calculate_pairwise_mi(cpts, var1, var2)
        
        return 0.0


    def recompute_cpt_with_new_structure(self, original_cpt, new_evidence, all_cpts):
        """
        Recompute CPT with new parent structure - fixed reshape issue
        """
        variable = original_cpt['variable']
        var_card = original_cpt['variable_card']
        
        # Calculate new evidence cardinalities
        new_evidence_cards = []
        for ev in new_evidence:
            ev_card = 2 
            for cpt in all_cpts:
                if cpt['variable'] == ev:
                    ev_card = cpt['variable_card']
                    break
                elif ev in cpt['evidence']:
                    idx = cpt['evidence'].index(ev)
                    ev_card = cpt['evidence_card'][idx]
                    break
            new_evidence_cards.append(ev_card)
        
        if new_evidence_cards:
            required_configs = np.prod(new_evidence_cards)
            new_shape = [var_card] + new_evidence_cards
        else:
            required_configs = 1
            new_shape = [var_card, 1]
        
        new_values = np.ones((var_card, required_configs)) / var_card
        
        # Try to preserve relationships from old CPT
        old_evidence = original_cpt['evidence']
        old_values = np.array(original_cpt['values'])
        
        # Flatten old values to 2D
        if old_values.ndim > 2:
            old_values = old_values.reshape(var_card, -1)
        elif old_values.ndim == 1:
            old_values = old_values.reshape(var_card, 1)
        elif old_values.ndim == 0:
            old_values = np.array([[old_values]])
        
        common_evidence = [ev for ev in new_evidence if ev in old_evidence]
        if len(common_evidence) == len(old_evidence) == len(new_evidence):
            # Same evidence set, use old values
            if old_values.shape[1] == required_configs:
                new_values = old_values
            else:
                marginal = np.mean(old_values, axis=1, keepdims=True)
                new_values = np.tile(marginal / np.sum(marginal), (1, required_configs))
        elif common_evidence:
            # Partial overlap, use marginal distribution
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
            else:
                pass
        
        return {
            'variable': variable,
            'variable_card': var_card,
            'values': new_values,
            'evidence': new_evidence,
            'evidence_card': np.array(new_evidence_cards, dtype=int),
            'cardinality': np.array([var_card] + new_evidence_cards, dtype=int)
        }


    def build_model_from_cpts(self, cpts: List[Dict[str, Any]]) -> DiscreteBayesianNetwork:
        """
        Build a complete Bayesian network model from conditional probability tables.
        
        Args:
            cpts: List of CPT dictionaries containing variable information
            
        Returns:
            A complete DiscreteBayesianNetwork with structure and parameters
        """
        model = DiscreteBayesianNetwork()
        
        variable_cardinality: Dict[str, int] = {}
        for cpt in cpts:
            variable_cardinality[cpt['variable']] = cpt['variable_card']
            for ev, ev_card in zip(cpt['evidence'], cpt['evidence_card']):
                if ev not in variable_cardinality:
                    variable_cardinality[ev] = ev_card
        
        model.add_nodes_from(variable_cardinality.keys())
        
        for cpt in cpts:
            child = cpt['variable']
            for parent in cpt['evidence']:
                if not model.has_edge(parent, child):
                    try:
                        model.add_edge(parent, child)
                    except ValueError as e:
                        if "loop" in str(e).lower():
                            print(f"Skipped edge {parent} -> {child} to avoid cycle")
                        else:
                            raise e
        
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
            
            # Marginalize out dropped parents
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
        
        return model