# artifact-fmcad24-docking
Tools and scripts for reproducing results for FMCAD 2024 paper on Neural Lyapunov Barrier Certificates to generate Reach while Avoid (RWA) Certificates. If memory issues arise, we recommend reducing the batch size during training.

Organization:
Each individual approach is documented and has relevant code contained in each folder. Additionally, a README is provided within each folder for code use and documentation.

Note that in this paper we only use the code in the "RWA with Safety", "FRWA with Safety", and "Compositional FRWA with Safety" folders for explaining our methodology and experiments. Still, we leave other folders with variations of these techniques, even if not described in the paper, to help showcase other related possibilities with our work.

Overall Libraries and Precursors:
This code in this repository requires Python 3. Additionally, to run code, Marabou and Gurobi should be installed as outlined in https://github.com/NeuralNetworkVerification/Marabou. Any other relevant libaries are in the requirements.txt file, and can be installed using the command: 

pip install -r requirements.txt