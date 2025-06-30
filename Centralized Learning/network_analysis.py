import pandas as pd
import json
import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, TreeSearch, PC, MmhcEstimator, MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.metrics import structure_score

import networkx as nx

from typing import List, Dict, Tuple, Any, Optional

import logging
logging.getLogger('pgmpy').setLevel(logging.WARNING)

class BayesianNetworkAnalyzer:
    """
    Performs comparative analysis of Bayesian Networks to identify the best model.
    """
    def __init__(self, dataset_path: str):
        """
        """
        self.dataset = pd.read_csv(dataset_path)
        self.models = {}
        self.scores = {}
        self.best_model = None
        self.best_estimator = None

        self.estimators = {
            'hc': 'Hill Climb Search',
            'ts': 'Tree Search',
            'pc': 'PC algorithm'
        }
    
    def build_model_hc(self, scoring_method: str = 'bic-d') -> Any:
        """
        Builds model using Hill Climb Search Algorithm.
        """
        try:           
            hc = HillClimbSearch(self.dataset)
            model = hc.estimate(scoring_method=scoring_method)
            return model
        except Exception as e:
            print(f"Hill Climb Failed! {str(e)}")
            return None
        
    def build_model_ts(self) -> Any:
        """
        Builds model using Tree Search Algorithm.
        """
        try:
            ts = TreeSearch(self.dataset)
            model = ts.estimate(class_node='class')
            return model
        except Exception as e:
            print(f"Tree Search Failed! {str(e)}")
            return None
        
    def build_model_pc(self, significance_level: float = 0.05) -> Any:
        """
        Builds model using PC Algorithm.
        """
        try:
            pc = PC(self.dataset)
            model = pc.estimate(significance_level=significance_level)
            return model
        except Exception as e:
            print(f"PC Algorithm failed: {str(e)}")
            return None
        
    def calculate_model_score(self, model: Any, scoring_method: str = 'bic-d') -> float:
        """
        Calculates model score based on the specified scoring method.
        """
        try:
            if model is None or len(model.edges()) == 0:
                return float('-inf')
            score = structure_score(model, self.dataset, scoring_method=scoring_method)
            return score
        except Exception as e:
            print(f"Score calculation failed: {str(e)}")
            return float('-inf')
        
    def build_models(self, scoring_method: str = 'bic-d', pc_significance: float=0.05) -> Dict[str, Any]:
        """
        Builds models with Hill Climb Search, Tree Search, and PC Algorithm.
        """
        print("Building models with the following estimators:\n")

        models = {}

        print("     - Hill Climb Search")
        models['hc'] = self.build_model_hc(scoring_method)

        print("\n   - Tree Search")
        models['ts'] = self.build_model_ts()

        print("\n   - PC Algorithm")
        models['pc'] = self.build_model_pc(pc_significance)

        print("\nModel Scores:")
        scores = {}
        for name, model in models.items():
            if model is not None:
                score = self.calculate_model_score(model, scoring_method)
                scores[name] = score
                print(f"  - {self.estimators.get(name, name)}: {score:.4f}")
            else:
                scores[name] = float('-inf')
                print(f"  - {self.estimators.get(name, name)}: Failed")

        self.models = models
        self.scores = scores

        return models

    def select_best_model(self) -> Tuple[str, Any]:
        """
        Selects the model with the highest score.
        """ 
        if not self.scores:
            raise ValueError("No models built yet. Please call build_all_models() to build your first model.")
        
        valid_scores = {k: v for k, v in self.scores.items() if v != float('-inf')}

        if not valid_scores:
            raise ValueError("No valid model found.")
        
        best_estimator = max(valid_scores, key=valid_scores.get)
        self.best_estimator = best_estimator
        self.best_model = self.models[best_estimator]

        print(f"\nBest performing model: {self.estimators.get(best_estimator, best_estimator)}")
        print(f"Score: {self.scores[best_estimator]:.4f}")

        return best_estimator, self.best_model
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Performs comparative analysis of all models and returns a dataframe of comparisons.
        """
        if not self.scores:
            raise ValueError("No models built yet. Please call build_all_models() to build your first model.")
        
        comparison_dataset = []
        for estimator, score in self.scores.items():
            
            model = self.models.get(estimator)
            if model is not None:
                edges = len(model.edges())
                nodes = len(model.nodes())
    
            else:
                edges = 0
                nodes = 0
            
            comparison_dataset.append({
                'Estimator': self.estimators.get(estimator, estimator),
                'Score': score if score != float('-inf') else 'Failed',
                'Edges': edges,
                'Nodes': nodes
            })
        
        df = pd.DataFrame(comparison_dataset)
       
        if 'Failed' in df['Score'].values:
            successful = df[df['Score'] != 'Failed'].copy()
            failed = df[df['Score'] == 'Failed'].copy()
            
            if not successful.empty:
                successful = successful.sort_values('Score', ascending=False)
                df = pd.concat([successful, failed], ignore_index=True)
        else:
            df = df.sort_values('Score', ascending=False)
        
        return df
        
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
            if self.best_model is None:
                raise ValueError("No best model selected. Please call select_best_model() to generate one.")
            model = self.best_model
        
        if hasattr(model, 'edges') and not hasattr(model, 'get_cpds'):
            try:
                bn_model = DiscreteBayesianNetwork(model.edges())
                bn_model.fit(self.dataset, estimator=MaximumLikelihoodEstimator)
                model = bn_model
                print("Converted DAG to BayesianNetwork and fitted with data")
                
            except Exception as e:
                print(f"Warning: Could not convert DAG to BayesianNetwork: {str(e)}")
                return []
        
        elif hasattr(model, 'get_cpds') and len(model.get_cpds()) == 0:
            try:
                model.fit(self.dataset, estimator=MaximumLikelihoodEstimator)
            except Exception as e:
                print(f"Warning: Could not fit the model parameters: {str(e)}")
                return []
        
        cpts_list = []
        try:
            for cpd in model.get_cpds():
                cpt_dict = {
                    "variable": cpd.variable,
                    "evidence": self.convert_to_python_object(cpd.get_evidence()) if cpd.get_evidence() else [],
                    "cardinality": self.convert_to_python_object(cpd.cardinality),
                    "values": self.convert_to_python_object(cpd.get_values())
                }
                cpts_list.append(cpt_dict)
        except Exception as e:
            print(f"Warning: Could not extract CPTs: {str(e)}")
        
        return cpts_list
    
    def save_cpts_to_json(self, filename: str = "cpts.json", model: Any = None) -> None:
        """
        Save CPTs of the best model to a JSON file.
        """
        cpts_list = self.extract_cpts(model)
        
        if cpts_list:
            with open(filename, "w") as f:
                json.dump(cpts_list, f, indent=4)
            print(f"CPTs saved to: {filename}")
        else:
            print("No CPTs to save.")

    def save_model_comparison(self, filename: str = "model_comparison.csv") -> None:
        """
        Saves the model comparison dataframe to a csv file.
        """
        comparison_df = self.get_model_comparison()
        comparison_df.to_csv(filename, index = False)
        print(f"\nModel comparison saved to: {filename}")

    def create_network_graph(self, model: Any = None) -> nx.Graph:
        """
        Returns the graph generated from a model.
        """
        if model is None:
            if self.best_model is None:
                raise ValueError("No best model selected. Please call select_best_model() to create one.")
            model = self.best_model

        graph = nx.Graph()
        if model is not None and len(model.edges()) > 0:
            graph.add_edges_from(model.edges())
        
        return graph
    
class NetworkVisualizer:
    """
    Generates network visualizations and comparative plots.
    """
    def __init__(self, analyzer: BayesianNetworkAnalyzer):
        self.analyzer = analyzer
    
    def plot_model_comparison(self, figsize: tuple = (10, 8),
                          save_path: Optional[str] = None) -> None:
        """
        Plots comparison of all models, highlighting the best ones.
        """
        comparison_df = self.analyzer.get_model_comparison()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
        all_algorithms = comparison_df['Estimator'].tolist()
        scores = []
        edges = []
        colors_score = []
        colors_edges = []
        
        valid_scores = [row['Score'] for _, row in comparison_df.iterrows() 
                    if row['Score'] != 'Failed']
        best_score = max(valid_scores) if valid_scores else None
        
        for idx, row in comparison_df.iterrows():
            if row['Score'] != 'Failed':
                scores.append(row['Score'])
                edges.append(row['Edges'])

                is_best = (row['Score'] == best_score)
                colors_score.append('#9f0000' if is_best else '#22666F')
                colors_edges.append('#22666F')
            else:
                scores.append(0) 
                edges.append(0)
                colors_score.append('#cccccc')  
                colors_edges.append('#cccccc')
        
 
        bars1 = ax1.bar(all_algorithms, scores, color=colors_score)
        ax1.set_title('Model Scores (Higher is Better)', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.tick_params(axis='x', rotation=90)
        
        for i, (bar, score) in enumerate(zip(bars1, scores)):
            if comparison_df.iloc[i]['Score'] != 'Failed':
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                        f'{score:.1f}',
                        ha='center', va='bottom', fontsize=9)
            else:
                ax1.text(bar.get_x() + bar.get_width()/2., 0.1,
                        'Failed',
                        ha='center', va='bottom', fontsize=9, color='red')

        bars2 = ax2.bar(all_algorithms, edges, color=colors_edges)
        ax2.set_title('Model Complexity (Number of Edges)', fontweight='bold')
        ax2.set_ylabel('Number of Edges')
        ax2.tick_params(axis='x', rotation=90)
        
        for bar, edge_count in zip(bars2, edges):
            if edge_count > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{edge_count}',
                        ha='center', va='bottom', fontsize=9)
        
        best_models = [row['Estimator'] for _, row in comparison_df.iterrows() 
                    if row['Score'] == best_score]
        
        if len(best_models) > 1:
            legend_text = f"Best Models (Score: {best_score:.1f}): {', '.join(best_models)}"
            ax1.legend([plt.Rectangle((0,0),1,1, facecolor='#9f0000')], 
                    [legend_text], loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1) 
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to: {save_path}")
        
        plt.show()
    
    def visualize_network(self, model: Any = None,
                      target_node: str = 'class',
                      target_color: str = '#9f0000',
                      node_color: str = '#22666F',
                      target_size: int = 1200,
                      node_size: int = 800,
                      seed: int = 23,
                      figsize: tuple = (10, 8),
                      save_path: Optional[str] = None) -> None:
        """
        Visualizes a single network structure.
        """

        if model is None:
            if not self.analyzer.scores:
                print("No models available to visualize.")
                return
                
            valid_scores = {k: v for k, v in self.analyzer.scores.items()
                        if v != float('-inf') and v != 'Failed'}
                        
            if not valid_scores:
                print("No valid models to visualize.")
                return
                
            best_estimator = max(valid_scores, key=valid_scores.get)
            model = self.analyzer.models.get(best_estimator)
            best_score = valid_scores[best_estimator]
                        
            if model is None:
                print("Best model is not available.")
                return
        else:
            best_estimator = None
            best_score = None
            for est, mdl in self.analyzer.models.items():
                if mdl is model:
                    best_estimator = est
                    best_score = self.analyzer.scores.get(est, 'Unknown')
                    break
                        
            if best_estimator is None:
                best_estimator = "Custom Model"
                best_score = "Unknown"

        graph = self.analyzer.create_network_graph(model)

        if len(graph.nodes()) == 0:
            print("No nodes available.")
            return

        fig, ax = plt.subplots(figsize=figsize)
        
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

        estimator_name = self.analyzer.estimators.get(best_estimator, best_estimator)
                    
        if isinstance(best_score, (int, float)) and best_score != float('-inf'):
            score_text = f"{best_score:.4f}"
        else:
            score_text = str(best_score)
                    
        title = f"Best estimator: {estimator_name}"
        if score_text != "Unknown":
            title += f" (Score: {score_text})"
                    
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to: {save_path}")

        plt.show()


    def plot_networks(self, figsize: tuple = (10,8), save_path: Optional[str] = None) -> None:
        """
        Plot network structures for all valid models, highlighting the best performing ones.
        """
        valid_models = {k: v for k,v in self.analyzer.models.items() if v is not None and len(v.edges()) > 0}
        if not valid_models:
            print("No valid models to visualize.")
            return

        valid_scores = {k: self.analyzer.scores[k] for k in valid_models.keys() 
                    if self.analyzer.scores[k] != float('-inf')}
        best_score = max(valid_scores.values()) if valid_scores else None
        
        best_models = [k for k, score in valid_scores.items() if score == best_score] if best_score else []
        
        num_models = len(valid_models)
        cols = min(3, num_models)
        rows = (num_models + cols-1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_models == 1:
            axes = [axes]
        elif rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        for idx, (estimator, model) in enumerate(valid_models.items()):
            ax = axes[idx] if num_models > 1 else axes[0]
            
            graph = self.analyzer.create_network_graph(model)
            pos = nx.spring_layout(graph, seed=23)
            
            is_best = estimator in best_models
        
            node_colors = ['#9f0000' if node == 'class' else '#22666F' for node in graph.nodes()]
            
            nx.draw(graph, pos, ax=ax, with_labels=True, 
                node_color=node_colors, node_size=600, 
                font_size=8, font_weight='bold')
            
            title = f"{self.analyzer.estimators.get(estimator, estimator)}"
            
            title += f"\nScore: {self.analyzer.scores[estimator]:.2f}, Edges: {len(model.edges())}"
            
            ax.set_title(title, fontweight='bold' if is_best else 'normal',
                        color='#9f0000' if is_best else 'black')
            
        
        for idx in range(num_models, len(axes)):
            axes[idx].set_visible(False)

        if len(best_models) > 1:
            fig.suptitle(f"Network Structures - Best Models: {', '.join([self.analyzer.estimators.get(m, m) for m in best_models])}", 
                        fontweight='bold', y=0.98)
        elif len(best_models) == 1:
            best_name = self.analyzer.estimators.get(best_models[0], best_models[0])
            fig.suptitle(f"Network Structures - Best Model: {best_name}", 
                        fontweight='bold', y=0.98)
        else:
            fig.suptitle("Network Structures", fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network structures visualization saved to: {save_path}")
        
        plt.show()


