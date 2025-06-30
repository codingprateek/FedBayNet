import os
import argparse
import traceback

from network_analysis import BayesianNetworkAnalyzer, NetworkVisualizer 

def main(data_path, output_dir):
    """
    Run the Bayesian network analysis pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Output Paths
    cpts_output = os.path.join(output_dir, "best_model_cpts.json")
    comparison_output = os.path.join(output_dir, "model_comparison.csv")
    comparison_plot = os.path.join(output_dir, "model_comparison.png")
    best_network_plot = os.path.join(output_dir, "best_network_structure.png")
    all_networks_plot = os.path.join(output_dir, "all_network_structures.png")

    try:
        analyzer = BayesianNetworkAnalyzer(data_path)
        analyzer.build_models(scoring_method='bic-d')

        best_estimator, best_model = analyzer.select_best_model()

        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        comparison_df = analyzer.get_model_comparison()
        print(comparison_df.to_string(index=False))

        analyzer.save_model_comparison(comparison_output)
        analyzer.save_cpts_to_json(cpts_output)

        visualizer = NetworkVisualizer(analyzer)
        visualizer.plot_model_comparison(save_path=comparison_plot)
        visualizer.visualize_network(save_path=best_network_plot)
        visualizer.plot_networks(save_path=all_networks_plot)

    except FileNotFoundError:
        print(f"Error: Could not find dataset file at {data_path}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Network Analysis on encoded dataset.")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the input dataset CSV file"
    )
    parser.add_argument(
        "--output", type=str, default="Results",
        help="Directory to store output results (default: Results)"
    )

    args = parser.parse_args()
    main(data_path=args.data, output_dir=args.output)
