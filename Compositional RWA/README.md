Compositional RWA certificate code files for reference and execution, without incorporating the nonlinear safe velocity constraint (instead opting for a flat 0.5 m/s absolute value velocity limit on each velocity component). We outline files and a brief description of included functions. We additionally present how to run code in order to execute our implementation.

For running code: Run run_cluster_exp_comb.py. Choose to modify lines 9 and 10 in order to change the positional limit (presented as absolute value). In order to change the initial controller file path, line 23 can be modified, and the previous model file can be set at line 21, and the previous positional bound limit can be set at line 22. Note that modifying the initialcontroller itself should only be done such that the architecture is consistent to the original "fixed_controller_20n_manhattan.pt" file.

run_cluster_exp_comb.py: File which executes code to learn a verified Compositional FRWA certificate and controller after various CEGIS iterations, given an initial controller file (in PyTorch format) trained using code in https://github.com/act3-ace/SafeRL in addition to the "design-for-verification" scheme.

training_exp_comb.py: File containing functions and classes for training the FRWA certificate and controller.

queries_comb.py: File containing verification functions using Marabou for run_cluster_exp_comb.py.

generate_combined_model_torch_comb.py: File containing helper functions for creating a singular ONNX network composed of relevant networks to provide to Marabou. Functions here are used in run_cluster_exp_comb.py.

attempt_conversion.py: File containing helper class for training_exp_comb.py, to ensure controller network architecture consistency.

convertsinglenetwork.py: File containing helper function to convert a singular PyTorch network to an ONNX network to provide to Marabou. Function here is used in run_cluster_exp_comb.py.