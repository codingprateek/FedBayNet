import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Any, List, Dict, Tuple, Set
from collections import defaultdict
import itertools

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, TreeSearch, PC, GES, BIC, MaximumLikelihoodEstimator
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
                        seed: int = 23,
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None,
                        title_prefix: str = "Bayesian Network") -> None:
        """
        Visualize a (directed) Bayesian network structure using NetworkX and Matplotlib.
        """
        graph = nx.DiGraph()
        
        if structure is not None:
            graph.add_edges_from(structure.edges())
        
        if graph.number_of_nodes() == 0:
            print("No nodes available for visualization.")
            return 
        
        _, ax = plt.subplots(figsize=figsize)
        
        pos = nx.spring_layout(graph, seed=seed, k=2, iterations=50)
        
        node_colors = [target_color if n == target_node else node_color for n in graph.nodes()]
        node_sizes  = [target_size if n == target_node else node_size  for n in graph.nodes()]
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
        nx.draw_networkx_labels(graph, pos,
                            labels={n: n for n in graph.nodes()},
                            font_color='black', font_weight='bold', ax=ax)
        
        nx.draw_networkx_edges(
            graph, pos,
            arrows=True,                     
            arrowstyle='-|>',               
            arrowsize=20,                   
            edge_color='gray',
            width=1.5,
            ax=ax
        )
        
        title = f"{title_prefix} - {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to: {save_path}")
        
        plt.show()

    def create_client_model(self, dataset: pd.DataFrame) -> DiscreteBayesianNetwork:
        df = dataset.dropna().reset_index(drop=True).copy()
        for c in df.columns:
            if not pd.api.types.is_categorical_dtype(df[c]):
                df[c] = df[c].astype("category")
        
        candidates = {}
        
        try:
            pc = PC(df)
            candidates["pc"] = pc.estimate(variant="stable", ci_test="chi_square", return_type="dag")
            print("PC algorithm completed successfully")
        except Exception as e:
            print(f"PC algorithm failed: {e}")
            pass
        
        try:
            hc = HillClimbSearch(df)
            candidates["hill_climb"] = hc.estimate(scoring_method='bic-d')
            print("Hill Climbing algorithm completed successfully")
        except Exception as e:
            print(f"Hill Climbing algorithm failed: {e}")
            pass
        
        try:
            hc_tabu = HillClimbSearch(df)
            candidates["tabu"] = hc_tabu.estimate(scoring_method='bic-d', tabu_length=100)
            print("Tabu Search algorithm completed successfully")
        except Exception as e:
            print(f"Tabu Search algorithm failed: {e}")
            pass
        
        try:
            tree = TreeSearch(df)
            candidates["chow_liu"] = tree.estimate(estimator_type="chow-liu")
            print("Chow-Liu Tree algorithm completed successfully")
        except Exception as e:
            print(f"Chow-Liu Tree algorithm failed: {e}")
            pass
        
        try:
            if 'class' in df.columns:
                tree_tan = TreeSearch(df)
                candidates["tan"] = tree_tan.estimate(estimator_type="tan", class_node="class")
                print("TAN (Tree Augmented Naive Bayes) algorithm completed successfully")
        except Exception as e:
            print(f"TAN algorithm failed: {e}")
            pass

        try:
            ges = GES(df)
            candidates["ges"] = ges.estimate(scoring_method='bic-d')
            print("GES (Greedy Equivalence Search) algorithm completed successfully")
        except Exception as e:
            print(f"GES algorithm failed: {e}")
            pass
        
        try:
            if df.shape[1] <= 8:
                from pgmpy.estimators import ExhaustiveSearch
                exhaustive = ExhaustiveSearch(df, scoring_method=BIC(df))
                candidates["exhaustive"] = exhaustive.estimate()
                print("Exhaustive Search algorithm completed successfully")
            else:
                print("Exhaustive Search skipped: too many variables (>8)")
        except Exception as e:
            print(f"Exhaustive Search algorithm failed: {e}")
            pass
        
        # Remove duplicate structures
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
            return empty_bn.fit(df, estimator=MaximumLikelihoodEstimator)
        
        # Evaluate all unique structures using BIC
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
            print("⚠ No structure could be evaluated - creating empty Bayesian Network")
            empty_bn = DiscreteBayesianNetwork()
            empty_bn.add_nodes_from(df.columns)
            return empty_bn.fit(df, estimator=MaximumLikelihoodEstimator)
        
        print(f"BEST ALGORITHM: {best_algorithm.upper()}")
        print(f"Best BIC Score: {best_score:.2f}")
        print(f"Network Structure: {len(best_structure.nodes())} nodes, {len(best_structure.edges())} edges")
        
        if len(best_structure.edges()) > 0:
            print(f"   Edges: {list(best_structure.edges())}")
        
        best_model = best_structure.fit(df, estimator=MaximumLikelihoodEstimator)
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


    def fuse_cpds(self, expert_cpd, data_cpd, expert_weight=0.6):
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
    
    def aggregate_cpts(self, clients_cpts: List[List[Dict[str, Any]]], dataset_sizes: List[float]):
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
                ev_map = dict(zip(ev, map(int, evc)))
                
                for p, c in ev_map.items():
                    parent_to_cards[p][int(c)] += weights[ci]

            sup_parents = sorted(parent_to_cards.keys())
            sup_parent_cards = {}
            for p in sup_parents:
                choices = parent_to_cards[p]
                chosen_card = max(choices.keys(), key=lambda card: choices[card])
                sup_parent_cards[p] = chosen_card
            sup_parent_card_list = [sup_parent_cards[p] for p in sup_parents]

            per_client = []
            for ci, cpd in chosen:
                ev_raw = cpd.get('evidence')
                ev = list(ev_raw) if ev_raw is not None else []
                evc_raw = cpd.get('evidence_card')
                evc_list = list(map(int, list(evc_raw) if evc_raw is not None else []))

                client_sel_supidx = [sup_parents.index(p) for p in ev]
                varc = int(cpd['variable_card'])
                values = cpd['values']

                per_client.append((ci, values, varc, evc_list, client_sel_supidx))

            table = {}
            if sup_parent_card_list:
                for idx_sup in itertools.product(*[range(c) for c in sup_parent_card_list]):
                    acc = [0.0] * var_card
                    wsum = 0.0
                    
                    for (ci, values, varc, evc_list, client_sel_supidx) in per_client:
                        subset = []
                        out_of_range = False
                        for k, sup_idx in enumerate(client_sel_supidx):
                            sup_val = idx_sup[sup_idx]
                            if sup_val >= evc_list[k]:
                                out_of_range = True
                                break
                            subset.append(sup_val)
                        if out_of_range:
                            continue

                        prob_vector = []
                        try:
                            for s in range(var_card):
                                v = values[s]
                                for idx in subset:
                                    v = v[idx]
                                prob_vector.append(float(v))
                        except:
                            if len(evc_list) == 0:
                                if isinstance(values, (int, float)):
                                    prob_vector = [float(values)] if var_card == 1 else [1.0/var_card] * var_card
                                elif hasattr(values, '__len__') and len(values) == var_card:
                                    prob_vector = [float(values[s]) for s in range(var_card)]
                                else:
                                    prob_vector = [1.0/var_card] * var_card
                            else:
                                values_array = np.array(values)
                                if values_array.ndim == 2:
                                    prod_evidence = 1
                                    for ec in evc_list:
                                        prod_evidence *= ec
                                    strides = []
                                    for i in range(len(evc_list)):
                                        stride = 1
                                        for j in range(i+1, len(evc_list)):
                                            stride *= evc_list[j]
                                        strides.append(stride)
                                    col = sum(idx * stride for idx, stride in zip(subset, strides))
                                    
                                    if col >= values_array.shape[1]:
                                        col = 0
                                    
                                    try:
                                        prob_vector = [float(values_array[s, col]) for s in range(var_card)]
                                    except:
                                        prob_vector = [1.0/var_card] * var_card
                                else:
                                    prob_vector = [1.0/var_card] * var_card

                        if len(prob_vector) != var_card:
                            continue
                            
                        w = weights[ci]
                        wsum += w
                        for i in range(var_card):
                            acc[i] += w * prob_vector[i]

                    if wsum > 0:
                        acc = [x / wsum for x in acc]
                    
                    s = sum(acc)
                    if s <= 0:
                        normalized_acc = [1.0 / len(acc)] * len(acc) if acc else []
                    else:
                        normalized_acc = [x / s for x in acc]
                    
                    table[idx_sup] = normalized_acc
            else:
                table[tuple()] = [1.0/var_card] * var_card

            if not sup_parent_card_list:
                nested_values = [table[tuple()][s] for s in range(var_card)]
            else:
                nested_values = []
                for s in range(var_card):
                    def build_nested(dim: int, prefix: Tuple[int, ...]):
                        if dim == len(sup_parent_card_list):
                            return table[prefix][s]
                        return [build_nested(dim+1, prefix + (i,)) for i in range(sup_parent_card_list[dim])]
                    nested_values.append(build_nested(0, tuple()))

            aggregated.append({
                'variable': var,
                'variable_card': var_card,
                'values': nested_values,
                'evidence': sup_parents,
                'evidence_card': sup_parent_card_list,
                'cardinality': [var_card] + sup_parent_card_list,
            })

        aggregated.sort(key=lambda d: d['variable'])
        return aggregated

    def get_edge_support_from_clients(self, parent, child, cpt_lists, weights):
        """
        Calculate weighted consensus for an edge across clients
        """
        total_weight = sum(weights)
        supporting_weight = 0.0
        
        for i, cpts in enumerate(cpt_lists):
            weight = weights[i]
            
            # Check if the client has parent -> child relationship
            child_cpt = next((c for c in cpts if c['variable'] == child), None)
            if child_cpt and parent in child_cpt.get('evidence', []):
                supporting_weight += weight
        
        return supporting_weight / total_weight if total_weight > 0 else 0.0

    def learn_parameters(self, cpt_lists, weights, max_changes=5, consensus_threshold=0.6):
        """
        Learn structure using client consensus with dependency strength analysis
        
        Args:
            cpt_lists: List of CPT lists from clients
            weights: Client weights for consensus
            max_changes: Maximum structural changes per iteration
            consensus_threshold: Minimum consensus required for edge consideration
        """
        # Start with CPTs aggregation
        current_cpts = self.aggregate_cpts(cpt_lists, weights)
        variables = [cpt['variable'] for cpt in current_cpts]
        all_variables = set(variables)
        
        print(f"Starting consensus-based learning with {len(current_cpts)} variables (threshold: {consensus_threshold})")
        
        current_edges = set()
        for cpt in current_cpts:
            child = cpt['variable']
            for parent in cpt['evidence']:
                current_edges.add((parent, child))
        
        changes_made = 0
        total_improvement = 0.0
        
        while changes_made < max_changes:
            best_change = None
            best_score = 0.01  # Minimum improvement threshold
            
            # Consider adding edges based on client consensus
            for parent in all_variables:
                for child in all_variables:
                    if parent != child and (parent, child) not in current_edges:
                        if self.would_create_cycle(parent, child, current_edges):
                            continue
                        
                        child_cpt = next((c for c in current_cpts if c['variable'] == child), None)
                        if child_cpt and len(child_cpt['evidence']) >= 4:  # Max 4 parents
                            continue
                        
                        edge_score = self.score_edge_from_client_consensus(parent, child, cpt_lists, weights, consensus_threshold)
                        
                        if edge_score > best_score:
                            memory_cost = self.estimate_edge_addition_cost(parent, child, current_cpts)
                            if memory_cost < 10000:  
                                best_change = ('add', (parent, child), edge_score)
                                best_score = edge_score
            
            # Consider removing edges with weak consensus
            for parent, child in current_edges:
                removal_score = self.score_edge_removal_consensus(parent, child, cpt_lists, weights, consensus_threshold)
                
                if removal_score > best_score:
                    best_change = ('remove', (parent, child), removal_score)
                    best_score = removal_score
            
            if best_change is None:
                print("No beneficial changes found, stopping")
                break
            
            action, edge, score = best_change
            parent, child = edge
            
            if action == 'add':
                current_cpts = self.add_edge_to_cpts(parent, child, current_cpts)
                current_edges.add((parent, child))
                print(f"Added edge {parent}->{child}, consensus score: {score:.4f}")
                
            elif action == 'remove':
                current_cpts = self.remove_edge_from_cpts(parent, child, current_cpts)
                current_edges.remove((parent, child))
                print(f"Removed edge {parent}->{child}, consensus score: {score:.4f}")
            
            changes_made += 1
            total_improvement += score
        
        print(f"Made {changes_made} changes, total improvement: {total_improvement:.4f}")
        return current_cpts

    def score_edge_from_client_consensus(self, parent, child, cpt_lists, weights, consensus_threshold):
        """
        Score edges based on client consensus and dependency strength
        """
        edge_support = self.get_edge_support_from_clients(parent, child, cpt_lists, weights)
        
        if edge_support < consensus_threshold:
            return 0.0
        
        # Calculate average dependency strength in supporting clients
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
        
        high_consensus_bonus = max(0.8, consensus_threshold + 0.2) 
        if edge_support > high_consensus_bonus:
            consensus_score *= 1.5
        
        return consensus_score

    def score_edge_removal_consensus(self, parent, child, cpt_lists, weights, consensus_threshold):
        """
        Score edge removal - higher score means removal is beneficial
        """
        # Low consensus for existing edge suggests it should be removed
        edge_support = self.get_edge_support_from_clients(parent, child, cpt_lists, weights)
        
        if edge_support > consensus_threshold: 
            return 0.0
        
        avg_weakness = 0.0
        total_weight = 0.0
        
        for i, cpts in enumerate(cpt_lists):
            weight = weights[i]
            child_cpt = next((c for c in cpts if c['variable'] == child), None)
            
            if child_cpt:
                if parent in child_cpt['evidence']:
                    # Measure weakness
                    strength = self.measure_dependency_strength(parent, child, child_cpt)
                    weakness = max(0, 0.5 - strength)  
                    avg_weakness += weight * weakness
                else:
                    avg_weakness += weight * 0.5
                
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        avg_weakness /= total_weight
        
        # Score for removal = weakness * (1 - consensus)
        removal_score = avg_weakness * (1.0 - edge_support)
        
        return removal_score

    def measure_dependency_strength(self, parent, child, child_cpt):
        """
        Measure how strongly child depends on parent in this CPT using information gain
        """
        values = np.array(child_cpt['values'])
        
        if parent not in child_cpt['evidence']:
            return 0.0
        
        try:
            parent_idx = child_cpt['evidence'].index(parent)
            
            if values.ndim == 1:
                return 0.1
            
            # Calculate marginal entropy of child
            if values.ndim == 2:
                marginal_child = np.mean(values, axis=1)
            else:
                # Multi-dimensional case - sum over all parent configurations
                axes_to_sum = tuple(range(1, values.ndim))
                marginal_child = np.sum(values, axis=axes_to_sum)
            
            marginal_child = marginal_child / (np.sum(marginal_child) + 1e-12)
            marginal_entropy = -np.sum(marginal_child * np.log2(marginal_child + 1e-12))
            
            # Calculate conditional entropy H(child|parent)
            conditional_entropy = 0.0
            
            if values.ndim == 2 and len(child_cpt['evidence']) == 1:
                # only one parent case
                parent_card = child_cpt['evidence_card'][0]
                parent_marginal = np.mean(values, axis=0)
                parent_marginal = parent_marginal / (np.sum(parent_marginal) + 1e-12)
                
                for p_val in range(parent_card):
                    if parent_marginal[p_val] > 1e-12:
                        conditional_dist = values[:, p_val]
                        conditional_dist = conditional_dist / (np.sum(conditional_dist) + 1e-12)
                        cond_ent = -np.sum(conditional_dist * np.log2(conditional_dist + 1e-12))
                        conditional_entropy += parent_marginal[p_val] * cond_ent
            else:
                # Multi-parent case 
                # Use average conditional entropy across parent values
                parent_card = child_cpt['evidence_card'][parent_idx] if parent_idx < len(child_cpt['evidence_card']) else 2
                
                total_configs = values.shape[1] if values.ndim == 2 else np.prod(values.shape[1:])
                configs_per_parent = total_configs // parent_card
                
                for p_val in range(parent_card):
                    start_idx = p_val * configs_per_parent
                    end_idx = min((p_val + 1) * configs_per_parent, total_configs)
                    
                    if values.ndim == 2:
                        slice_values = values[:, start_idx:end_idx]
                        conditional_dist = np.mean(slice_values, axis=1)
                    else:
                        flat_values = values.reshape(values.shape[0], -1)
                        conditional_dist = np.mean(flat_values[:, start_idx:end_idx], axis=1)
                    
                    conditional_dist = conditional_dist / (np.sum(conditional_dist) + 1e-12)
                    cond_ent = -np.sum(conditional_dist * np.log2(conditional_dist + 1e-12))
                    conditional_entropy += cond_ent / parent_card  # Equal weighting
            
            # Information gain 
            info_gain = marginal_entropy - conditional_entropy
            max_possible_gain = marginal_entropy
            normalized_gain = info_gain / (max_possible_gain + 1e-12)
            
            return max(0.0, min(1.0, normalized_gain))
            
        except Exception as e:
            print(f"Warning: Error measuring dependency strength for {parent}->{child}: {e}")
            return 0.0

    def get_variable_cardinality(self, variable, cpts):
        """
        Get cardinality of a variable from CPTs
        """
        for cpt in cpts:
            if cpt['variable'] == variable:
                return cpt['variable_card']
            elif variable in cpt.get('evidence', []):
                idx = cpt['evidence'].index(variable)
                if idx < len(cpt['evidence_card']):
                    return cpt['evidence_card'][idx]
        
        return 2  
    
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
            # Fallback: simple path check
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

    def recompute_cpt_with_new_structure(self, original_cpt, new_evidence, all_cpts):
        """
        Recompute CPT with new parent structure - fixed reshape issue
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