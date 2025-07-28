import os
import yaml
import pandas as pd
import json
from FeatureCloud.app.engine.app import AppState, Role, app_state, LogLevel, State
from bn_learning import Client, Coordinator
import shutil


@app_state('initial', Role.BOTH)
class InitialState(AppState):
    def register(self):
        self.register_transition('read_input', Role.BOTH)

    def run(self):
        self.log("[CLIENT] Starting FedBayNet...")
        self.log(f"[CLIENT] Node ID: {self.id}, Coordinator: {self.is_coordinator}")
        return 'read_input'


@app_state('read_input', Role.BOTH)
class ReadInputState(AppState):
    def register(self):
        self.register_transition('local_computation', Role.BOTH)
        self.register_transition('read_input', Role.BOTH)

    def run(self) -> str | None:
        try:
            self.log("[CLIENT] Reading dataset and configuration")
            self.read_config()

            splits = self.load('splits')
            models = self.load('models')

            for split_path in splits.keys():
                models[split_path] = Coordinator() if self.is_coordinator else Client()

                train_file = self.load('dataset')
                train_path = os.path.join(split_path, train_file)

                if not os.path.exists(train_path):
                    self.log(f"[CLIENT] Error: Training file not found at {train_path}")
                    raise FileNotFoundError(f"Training file not found: {train_path}")

                data = pd.read_csv(train_path, sep=self.load('sep'))
                splits[split_path] = data
                self.log(f"[CLIENT] Loaded data for split {split_path}: {data.shape[0]} rows, {data.shape[1]} columns")

            self.store('models', models)
            self.store('splits', splits)

            client_split_path = None
            client_id_str = str(self.id).lower()

            for split_path in splits.keys():
                split_dirname = os.path.basename(split_path).lower()
                self.log(f"[CLIENT] Comparing client id '{client_id_str}' with split directory '{split_dirname}'")
                if client_id_str in split_dirname:
                    client_split_path = split_path
                    break

            if client_split_path is None:
                if len(splits) == 1:
                    client_split_path = next(iter(splits.keys()))
                    self.log(f"[CLIENT] No exact client id match; using only split directory: {client_split_path}")
                else:
                    raise RuntimeError(f"[CLIENT {self.id}] No matching split directory found for this client.")

            self.store('dataset', splits[client_split_path])
            self.store('client_split_path', client_split_path)

            # Initialize the client instance
            client = models[client_split_path]
            self.store('client_instance', client)

            return 'local_computation'

        except Exception as e:
            self.log(f'[CLIENT] Error reading config: {str(e)}')
            self.update(message='no config file or missing fields', state=State.ERROR)
            import traceback
            traceback.print_exc()
            return 'read_input'

    def read_config(self):
        input_dir = "/mnt/input"
        output_dir = "/mnt/output"

        self.store('INPUT_DIR', input_dir)
        self.store('OUTPUT_DIR', output_dir)

        config_path = os.path.join(input_dir, 'config.yml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            full_config = yaml.load(f, Loader=yaml.FullLoader)

        config = full_config['fc_fedbaynet']

        self.store('dataset', config['input']['dataset'])
        self.store('sep', config['format']['sep'])
        self.store('split_mode', config['split']['mode'])
        self.store('split_dir', config['split']['dir'])

        splits = {}
        if self.load('split_mode') == "directory":
            split_base_dir = os.path.join(input_dir, self.load('split_dir'))
            if os.path.exists(split_base_dir):
                splits = {f.path: None for f in os.scandir(split_base_dir) if f.is_dir()}
            else:
                splits = {input_dir: None}
        else:
            splits = {input_dir: None}

        models = {k: None for k in splits.keys()}

        for split_path in splits.keys():
            output_path = split_path.replace("/input/", "/output/")
            os.makedirs(output_path, exist_ok=True)

        output_config = os.path.join(output_dir, 'config.yml')
        shutil.copyfile(config_path, output_config)

        self.log('[CLIENT] Configuration loaded successfully')

        self.store('models', models)
        self.store('splits', splits)


@app_state('local_computation', Role.BOTH)
class LocalComputationState(AppState):
    def register(self):
        self.register_transition('aggregation', Role.COORDINATOR)
        self.register_transition('await_completion', Role.PARTICIPANT)

    def run(self):
        self.log("[CLIENT] Starting local Bayesian Network learning")

        dataset = self.load('dataset')
        if dataset is None:
            raise RuntimeError(f"[CLIENT {self.id}] Dataset not found for this client.")

        client = self.load('client_instance')
        if client is None:
            client = Client()
            self.store('client_instance', client)

        # Perform local learning
        result = client.learn_local_structure(dataset)
        size = result["size"]
        local_cpts = result["cpts"]

        # Save local CPTs to JSON
        output_dir = self.load('OUTPUT_DIR')
        os.makedirs(output_dir, exist_ok=True)
        local_cpts_file = os.path.join(output_dir, "local_cpts.json")
        client.save_cpts_to_json(local_cpts_file)

        # Visualize and save local network
        local_network_image = os.path.join(output_dir, "local_network.png")
        client.visualize_network(
            save_path=local_network_image,
            title_prefix="Local Bayesian Network"
        )

        payload = {"size": size, "cpts": local_cpts}
        self.send_data_to_coordinator(payload)

        if self.is_coordinator:
            return 'aggregation'
        else:
            return 'await_completion'


@app_state('await_completion', Role.PARTICIPANT)
class AwaitCompletionState(AppState):
    def register(self):
        self.register_transition('final', Role.PARTICIPANT)

    def run(self):
        self.log("[CLIENT] Waiting for coordinator to complete aggregation...")
        
        # Wait for completion signal from coordinator
        completion_data = self.await_data()
        
        if completion_data.get("status") == "completed":
            self.log("[CLIENT] Received completion signal from coordinator.")
            if completion_data.get("global_network_saved"):
                self.log("[CLIENT] Global network visualization was successfully saved by coordinator")
            else:
                self.log("[CLIENT] Warning: Global network visualization may not have been saved")
        else:
            self.log("[CLIENT] Warning: Unexpected completion signal received", LogLevel.WARNING)
        
        return 'final'


@app_state('aggregation', Role.COORDINATOR)
class AggregationState(AppState):
    def register(self):
        self.register_transition('broadcast_results', Role.COORDINATOR)

    def run(self):
        self.log("[COORDINATOR] Aggregating CPTs from all clients...")

        received = self.gather_data()
        client_sizes = [d["size"] for d in received]
        client_cpts = [d["cpts"] for d in received]

        coordinator = self.load('client_instance')
        if coordinator is None:
            coordinator = Coordinator()
            self.store('client_instance', coordinator)

        # Aggregate CPTs
        global_cpts = coordinator.aggregate_cpts(client_cpts, client_sizes)

        self.log(f"[COORDINATOR] Aggregated CPTs for {len(global_cpts)} variables")
        
        # Save global CPTs to JSON
        output_dir = self.load('OUTPUT_DIR')
        os.makedirs(output_dir, exist_ok=True)
        global_cpts_file = os.path.join(output_dir, "global_cpts.json")
        
        try:
            with open(global_cpts_file, 'w') as f:
                json.dump(global_cpts, f, indent=4)
            self.log(f"[COORDINATOR] Saved global CPTs to {global_cpts_file}")
        except Exception as e:
            self.log(f"[COORDINATOR] Error saving global CPTs: {e}")

        # Create and visualize global network
        global_model = coordinator.create_global_network_from_cpts(global_cpts)
        global_network_saved = False
        
        if global_model:
            try:
                global_network_image = os.path.join(output_dir, "global_network.png")
                coordinator.visualize_network(
                    model=global_model,
                    save_path=global_network_image,
                    title_prefix="Global Federated Bayesian Network"
                )
                global_network_saved = True
                self.log(f"[COORDINATOR] Saved global network visualization to {global_network_image}")
            except Exception as e:
                self.log(f"[COORDINATOR] Error saving global network visualization: {e}")
        else:
            self.log("[COORDINATOR] Warning: Could not create global network model")
        
        # Store global results for final state
        self.store('global_cpts', global_cpts)
        self.store('global_model', global_model)
        self.store('global_network_saved', global_network_saved)
        
        return 'broadcast_results'


@app_state('broadcast_results', Role.COORDINATOR)
class BroadcastResultsState(AppState):
    def register(self):
        self.register_transition('final', Role.COORDINATOR)

    def run(self):
        self.log("[COORDINATOR] Broadcasting completion signal to all participants...")
        
        # Send completion signal to all participants
        global_cpts = self.load('global_cpts')
        num_variables = len(global_cpts) if global_cpts else 0
        
        completion_data = {
            "status": "completed",
            "global_network_saved": self.load('global_network_saved'),
            "num_variables": num_variables
        }
        
        self.broadcast_data(completion_data)
        self.log("[COORDINATOR] Completion signal broadcasted.")
        
        return 'final'


@app_state('final', Role.BOTH)
class FinalState(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self) -> str | None:
        self.log("[CLIENT] Learning Complete.")
        
        try:
            output_dir = self.load('OUTPUT_DIR')
            
            if self.is_coordinator:
                global_cpts = self.load('global_cpts')
                global_model = self.load('global_model')
                global_network_saved = self.load('global_network_saved')
                
                if global_cpts:
                    self.log(f"[COORDINATOR] Final global model contains {len(global_cpts)} variables")
                    
                    # Save final summary
                    summary = {
                        "num_variables": len(global_cpts),
                        "variables": [cpt["variable"] for cpt in global_cpts],
                        "total_edges": len(global_model.edges()) if global_model else 0,
                        "model_structure": list(global_model.edges()) if global_model else [],
                        "global_network_saved": global_network_saved
                    }
                    
                    summary_file = os.path.join(output_dir, "federated_learning_summary.json")
                    with open(summary_file, "w") as f:
                        json.dump(summary, f, indent=4)
                    
                    self.log(f"[COORDINATOR] Saved final summary to {summary_file}")
                    
                else:
                    self.log("[COORDINATOR] Warning: No global CPTs found")
            else:
                client = self.load('client_instance')
                if client and client.model:
                    local_cpts = client.extract_cpts()
                    self.log(f"[CLIENT] Final local model contains {len(local_cpts)} variables")
                else:
                    self.log("[CLIENT] Warning: No local model found")
                    
        except Exception as e:
            self.log(f"[CLIENT] Error in final state: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return 'terminal'