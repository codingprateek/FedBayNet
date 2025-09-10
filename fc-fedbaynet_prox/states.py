import os
import yaml
import numpy as np
import pandas as pd
import json
from FeatureCloud.app.engine.app import AppState, Role, app_state, State
import numpy as np
import traceback
from algorithm import Client, Coordinator
import matplotlib.pyplot as plt
import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)

# FedBayNet States
INITIAL = 'initial'
READ_INPUT = 'read input'
LOCAL_COMPUTATION = 'local computation'
AGGREGATION = 'aggregation'
AWAIT_AGGREGATION = 'await aggregation'
FINAL = 'final'
TERMINAL = 'terminal'


@app_state(INITIAL, Role.BOTH)
class InitialState(AppState):
    """
    Initializes FedBayNet.
    """
    def register(self):
        self.register_transition(READ_INPUT, Role.BOTH)

    def run(self):
        self.log("[CLIENT] Starting FedBayNet...")
        self.log(f"[CLIENT] Node ID: {self.id}, Coordinator: {self.is_coordinator}")
        self.log("Initial to Read Input")
        return READ_INPUT


@app_state(READ_INPUT, Role.BOTH)
class ReadInputState(AppState):
    """
    Reads client datasets, blacklists and whitelists, along with a common configuration file.
    """
    def register(self):
        self.register_transition(LOCAL_COMPUTATION, Role.BOTH)
        self.register_transition(READ_INPUT, Role.BOTH)

    def run(self):
        try:
            self.log("[CLIENT] Reading dataset and configuration...")
            self.read_config()

            splits = self.load('splits')
            roles = self.load('roles')

            for split_path in splits.keys():
                roles[split_path] = Coordinator() if self.is_coordinator else Client()

                dataset_loc = self.load('dataset')
                dataset_path = os.path.join(split_path, dataset_loc)
                if not os.path.exists(dataset_path):
                    raise FileNotFoundError(f"Dataset File not found at location: {dataset_path}")
                
                dataset = pd.read_csv(dataset_path)
                splits[split_path] = dataset
                self.log(f"[CLIENT] Loaded dataset from {split_path}: {dataset.shape[0]} records and {dataset.shape[1]} variables")

                bwlists_loc = self.load('bwlists')
                bwlists_path = os.path.join(split_path, bwlists_loc)
                if not os.path.exists(bwlists_path):
                    raise FileNotFoundError(f"Dataset File not found at location: {bwlists_path}")
                
                with open(bwlists_path) as f:
                    expknowledge = json.load(f)

                blacklist = [tuple(edge) for edge in expknowledge["blacklist"]]
                whitelist = [tuple(edge) for edge in expknowledge["whitelist"]]

            self.store('roles', roles)
            self.store('splits', splits)

            client_split_path = None
            client_id_str = str(self.id).lower()

            for split_path in splits.keys():
                split_dirname = os.path.basename(split_path).lower()
                self.log(f"[CLIENT] Comparing Client ID '{client_id_str}' with split directory '{split_dirname}'")
                if client_id_str in split_dirname: # FLAG: change it later for client folders
                    client_split_path = split_path
                    break 

            if client_split_path is None:
                if len(splits) == 1:
                    client_split_path = next(iter(splits.keys()))
                    self.log(f"[CLIENT] Using split directory: {client_split_path}")
                else:
                    raise RuntimeError(f"[CLIENT {self.id}]: No matching split directory found for this client.")
                
            self.store('dataset', splits[client_split_path])
            self.store('blacklist', blacklist)
            self.store('whitelist', whitelist)
            self.store('client_split_path', client_split_path)

            self.log(f"BLACKLIST: {blacklist}")
            self.log(f"WHITELIST: {whitelist}")
            client = roles[client_split_path]
            self.store('client_instance', client)
            self.store('iteration', 1)
            self.store('accuracy_history', [])

            return LOCAL_COMPUTATION
        
        except Exception as e:
            self.log(f"[CLIENT] Error reading config: {str(e)}")
            self.update(message = "no config file or missing fields", state = State.ERROR)
            traceback.print_exc()
            return READ_INPUT

    def read_config(self):
        self.log("Read Input to Local Computation")

        input_dir = "/mnt/input"
        output_dir = "/mnt/output"

        self.store('input_dir', input_dir)
        self.store('output_dir', output_dir)

        config_path = os.path.join(input_dir, 'config.yml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config_file = yaml.load(f, Loader=yaml.FullLoader)

        config = config_file['fc_fedbaynet_prox']
        self.store('dataset', config['input']['dataset_loc'])
        self.store('bwlists', config['input']['bwlists_loc'])
        self.store('split_mode', config['split']['mode'])
        self.store('split_dir', config['split']['dir']) 
        self.store('max_iterations', config['max_iterations'])
        self.store('cv_folds', config['cv_folds'])

        self.store('mu', config['fedprox']['mu'])
        self.store('epochs', config['fedprox']['epochs'])
        self.store('lr', config['fedprox']['lr'])

        self.store('expert_weight', config['network_fusion']['expert_weight'])
        self.store('add_node_threshold', config['network_fusion']['add_node_threshold'])
        self.store('add_edge_threshold', config['network_fusion']['add_edge_threshold'])
        self.store('reverse_edge_threshold', config['network_fusion']['reverse_edge_threshold'])
        self.store('remove_edge_threshold', config['network_fusion']['remove_edge_threshold'])
        self.store('max_changes_fusion', config['network_fusion']['max_changes_fusion'])

        self.store('addition_threshold', config['consensus_params']['addition_threshold'])
        self.store('removal_threshold', config['consensus_params']['removal_threshold'])
        self.store('reversal_threshold', config['consensus_params']['reversal_threshold'])
        self.store('node_addition_threshold', config['consensus_params']['node_addition_threshold'])
        self.store('max_changes', config['consensus_params']['max_changes'])

        
        splits = {}
        if self.load('split_mode') == "directory":
            split_base_dir = os.path.join(input_dir, self.load('split_dir'))
            if os.path.exists(split_base_dir):
                splits = {f.path: None for f in os.scandir(split_base_dir) if f.is_dir()}
            else:
                splits = {input_dir: None}
        else:
            splits = {input_dir: None}

        roles = {}
        for split_path in splits.keys():
            output_path = split_path.replace("/input/", "/output/")
            os.makedirs(output_path, exist_ok=True)

        self.log("[CLIENT] Configuration Loaded Successfully")

        self.store('roles', roles)
        self.store('splits', splits)


@app_state(LOCAL_COMPUTATION, Role.BOTH)
class LocalComputationState(AppState):
    """
    Accessible by clients and coordinator as a client for:
    - creating client model based on dataset
    - fusing local model with global network to incorportate expert and dataset knowledge to compute CPTs
    - evaluating fused model on local dataset using K-fold Cross Validation

    The client payload sent to the coordinator after this step contains:
    - [Iteration 1] client's blacklist, whitelist and iteration value 
    - [Iteration > 1] client's CPTs, dataset size and iteration value
    """
    def register(self):
        self.register_transition(AGGREGATION, Role.COORDINATOR)
        self.register_transition(AWAIT_AGGREGATION, Role.PARTICIPANT)

    def run(self):
        output_dir = "/mnt/output"
        iteration = self.load('iteration')
        max_iterations = self.load('max_iterations')
        dataset = self.load('dataset')
        blacklist = self.load('blacklist')
        whitelist = self.load('whitelist')
        expert_weight = self.load('expert_weight')
        add_node_threshold = self.load('add_node_threshold')
        add_edge_threshold = self.load('add_edge_threshold')
        reverse_edge_threshold = self.load('reverse_edge_threshold')
        remove_edge_threshold = self.load('remove_edge_threshold')
        max_changes_fusion = self.load('max_changes_fusion')

        mu = self.load('mu')
        epochs = self.load('epochs')
        lr = self.load('lr')
        
        participant = Coordinator() if self.is_coordinator else Client()

        if iteration == 1:
            self.log(f"[CLIENT] Initializing FedBayNet...")
            self.log(f"[CLIENT] Sharing expert knowledge with the coordinator.")
            client_payload = {
                "blacklist":  blacklist,
                "whitelist": whitelist,
                "iteration": iteration
            }
            initial_local_network = participant.create_client_model(dataset)
            self.store('initial_local_network', initial_local_network)
        else:
            if self.is_coordinator:
                global_network = self.load("global_network")
                aggregated_cpts = self.load('aggregated_cpts')
            else:
                global_network = self.load("global_network_client")
                aggregated_cpts = self.load("aggregated_cpts_client")
            
            initial_local_network = self.load('initial_local_network')
            client_network = participant.fuse_bayesian_networks(global_network, initial_local_network, dataset, 
                                                                expert_weight=expert_weight, 
                                                                add_node_threshold=add_node_threshold, 
                                                                add_edge_threshold=add_edge_threshold, 
                                                                reverse_edge_threshold=reverse_edge_threshold, 
                                                                remove_edge_threshold=remove_edge_threshold,
                                                                max_changes = max_changes_fusion)

            local_cpts = participant.compute_local_cpts_from_structure_and_data(client_network, dataset)
            client_cpts = participant.fedprox_local_update(local_cpts, aggregated_cpts, dataset, mu=mu, epochs=epochs, lr=lr)

            if iteration > max_iterations:
                results_path = os.path.join(output_dir, "results.csv")
                avg_acc = participant.kfold_cv(dataset, global_network, csv_filename=results_path)
                self.log(f"[CLIENT] Final results saved to {results_path}")
            else:
                avg_acc = participant.kfold_cv(dataset, global_network, csv_filename=None)
                
            self.log(f"[CLIENT] Average Accuracy for Global Network {iteration}: {avg_acc}")
            
            accuracy_history = self.load('accuracy_history')
            accuracy_history.append(avg_acc)
            self.store("accuracy_history", accuracy_history)
                    
            client_payload = {
                "client_cpts": client_cpts,
                "dataset_size": dataset.shape[0],
                "iteration": iteration
            }

            visualization_path = os.path.join(output_dir, f"global_network_iter{iteration-1}.png")
            participant.visualize_network(global_network, save_path=visualization_path)
            local_visualization_path = os.path.join(output_dir, f"local_network_iter{iteration-1}.png")
            participant.visualize_network(client_network, save_path=local_visualization_path)

        
        self.send_data_to_coordinator(client_payload)

        if self.is_coordinator:
            return AGGREGATION
        else:
            return AWAIT_AGGREGATION
        

@app_state(AGGREGATION, Role.COORDINATOR)
class AggregateState(AppState):
    """
    Accessible only by the coordinator(server) to:
    - build initial global network using expert knowledge sent by clients, i.e., respective blacklists and whitelists.
    - aggregate client CPTs using weighted averaging
    - build a Bayesian network using the aggregated CPTs

    The coordinator payload broadcasted to clients after this state contains:
    1. Global network 
    2. Aggregated CPTs
    3. Iteration value
    4. A message to indicate termination of learning process to the clients
    """
    def register(self):
        self.register_transition(LOCAL_COMPUTATION, Role.COORDINATOR)
        self.register_transition(FINAL, Role.COORDINATOR)

    def convert_np_arrays(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: self.convert_np_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self.convert_np_arrays(i) for i in obj]
            else:
                return obj
                
    def run(self):
        output_dir = "/mnt/output"
        iteration = self.load('iteration')
        max_iterations = self.load('max_iterations')
        addition_threshold = self.load('addition_threshold')
        removal_threshold = self.load('removal_threshold')
        reversal_threshold = self.load('reversal_threshold')
        node_addition_threshold = self.load('node_addition_threshold')
        max_changes = self.load('max_changes')

        self.log(f"[COORDINATOR] Aggregating iteration {iteration}")

        coordinator = Coordinator()
        client_payload = self.gather_data()
        prev_aggregated_cpts = self.load('aggregated_cpts')

        if iteration == 1:
            self.log("[COORDINATOR] Building network structure based on expert knowledge...")
            blacklists = [payload["blacklist"] for payload in client_payload]
            whitelists = [payload["whitelist"] for payload in client_payload]
            aggregated_cpts = []
            global_network = coordinator.create_constrained_global_network(whitelists, blacklists)
        else:
            self.log("[COORDINATOR] Aggregating CPTs and building global network structure...")
            clients_cpts = [payload["client_cpts"] for payload in client_payload]
            dataset_sizes = [payload["dataset_size"] for payload in client_payload]
            aggregated_cpts = coordinator.learn_parameters(
                                clients_cpts, 
                                dataset_sizes, 
                                max_changes=max_changes, 
                                addition_threshold=addition_threshold,
                                removal_threshold=removal_threshold, 
                                reversal_threshold=reversal_threshold,
                                node_addition_threshold=node_addition_threshold
                            )
            global_network = coordinator.build_model_from_cpts(aggregated_cpts)

        self.store("aggregated_cpts", aggregated_cpts)
        self.store("global_network", global_network)

        os.makedirs(output_dir, exist_ok=True)
        serialized_cpts = self.convert_np_arrays(aggregated_cpts)
        cpts_path = os.path.join(output_dir, f"aggregated_cpts_{iteration}.json")
        with open(cpts_path, "w") as json_file:
            json.dump(serialized_cpts, json_file, indent=4)

        self.log(f"[DEBUG] Saved CPTs to: {cpts_path}")
        self.log(f"[DEBUG] Aggregated CPT count: {len(aggregated_cpts)}")

        # Stopping condition
        if iteration > int(max_iterations):
            self.broadcast_data({'message': 'done'})
            self.log(f"[COORDINATOR] Completed after {iteration} iterations (max iterations: {max_iterations})")
            return FINAL

        iteration += 1
        self.store('iteration', iteration)
        coordinator_payload = {
            "global_network": global_network,
            "aggregated_cpts": serialized_cpts,
            "iteration": iteration,
            "message": "continue"
        }
        self.broadcast_data(coordinator_payload)
        self.log(f"[COORDINATOR] Broadcasting for iteration {iteration} (max iterations: {max_iterations})")
        return LOCAL_COMPUTATION



@app_state(AWAIT_AGGREGATION, Role.PARTICIPANT)
class AwaitAggregationState(AppState):
    """
    Clients transition into this state to wait for aggregation results from the coordinator.
    """
    def register(self):
        self.register_transition(LOCAL_COMPUTATION, Role.PARTICIPANT)
        self.register_transition(FINAL, Role.PARTICIPANT)

    def run(self):
        coordinator_payload = self.await_data()
        message = coordinator_payload.get('message')
        global_network_client = coordinator_payload.get("global_network")
        self.store("global_network_client", global_network_client)
        aggregated_cpts_client = coordinator_payload.get("aggregated_cpts")
        self.store("aggregated_cpts_client", aggregated_cpts_client)
        
        if message == 'done':
            self.log(f"[CLIENT] Received completion signal")
            return FINAL
        
        new_iteration = coordinator_payload.get('iteration')
        if new_iteration is not None:
            self.store('iteration', new_iteration)
            self.log(f"[CLIENT] Updated to iteration {new_iteration}")

        return LOCAL_COMPUTATION

    
@app_state(FINAL, Role.BOTH)
class FinalState(AppState):
    def register(self):
        self.register_transition(TERMINAL, Role.BOTH)

    def plot_accuracy(self, accs, output_dir, filename="accuracy_trend.png"):
        os.makedirs(output_dir, exist_ok=True)

        if not accs or len(accs) == 0:
            print("No accuracy data to plot")
            return None
        
        accs = np.clip(accs, 0.0, 1.0)
        iterations = np.arange(1, len(accs) + 1)

        plt.figure(figsize=(10, 6))
        
        plt.plot(iterations, accs, marker='o', linestyle='-', 
                color='#22666F', linewidth=2, markersize=6, 
                markerfacecolor='#22666F', markeredgecolor='white', 
                markeredgewidth=1)
        
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Average Accuracy", fontsize=12)
        
        max_iterations = len(accs)
        if max_iterations <= 15:
            plt.xticks(iterations)
        else:
            tick_interval = max(1, max_iterations // 10)
            tick_positions = np.arange(1, max_iterations + 1, tick_interval)
            if max_iterations not in tick_positions:
                tick_positions = np.append(tick_positions, max_iterations)
            plt.xticks(tick_positions)
        
        min_acc = min(accs)
        max_acc = max(accs)
        
        if max_acc >= 0.99:
            plt.ylim(max(0.0, min_acc - 0.05), 1.02)
            plt.yticks(np.arange(max(0.0, min_acc - 0.05), 1.05, 0.05))
        elif min_acc <= 0.01:
            plt.ylim(-0.02, max(1.0, max_acc + 0.05))
            plt.yticks(np.arange(0.0, max(1.0, max_acc + 0.05), 0.1))
        else:
            buffer = (max_acc - min_acc) * 0.1 + 0.02
            plt.ylim(max(0.0, min_acc - buffer), min(1.02, max_acc + buffer))
            plt.yticks(np.arange(0.0, 1.05, 0.1))
        
        plt.grid(True, alpha=0.3, linestyle='--')
        
        if len(accs) <= 15:
            for i, acc in enumerate(accs):
                if i == 0 or i == len(accs) - 1 or acc >= 0.99 or acc <= 0.01:
                    plt.annotate(f'{acc:.3f}', 
                            (iterations[i], acc),
                            textcoords="offset points", 
                            xytext=(0, 10), 
                            ha='center',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='yellow', 
                                    alpha=0.7,
                                    edgecolor='none'))
        
        stats_text = f'Final: {accs[-1]:.3f}\nBest: {max_acc:.3f}\nWorst: {min_acc:.3f}'
        plt.text(0.98, 0.02, stats_text, 
                transform=plt.gca().transAxes, 
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='lightblue', 
                        alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        acc_plot_path = os.path.join(output_dir, filename)
        plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        return acc_plot_path

    def run(self):
        self.log("Learning Complete")

        accs = self.load("accuracy_history")
        output_dir = "/mnt/output"
        acc_plot_path = self.plot_accuracy(accs, output_dir)

        self.log(f"[FINAL] Accuracy trend saved to {acc_plot_path}")
        return TERMINAL