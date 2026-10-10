## Mass-Conserving Perceptron (MCP) Neural Network Model Zoo

This repository provides a curated and documented implementation of the neural-network architectures developed in Wang & Gupta (2025), all built around the mass-conserving perceptron (MCP) as the fundamental computational unit, as illustrated in Figure 1.

![MCP-based neural-network architecture](Figure1.png)

*Figure 1. A neural-network architecture built from mass-conserving perceptrons.*

The original research code was archived at [Zenodo](https://doi.org/10.5281/zenodo.15811866); the version presented here has been subsequently streamlined, refactored, and reorganized for improved readability, reproducibility, and reuse, with assistance from OpenAI's ChatGPT (GPT-6.1 Sol, Max intelligence mode).

Note that this repository includes only the results for the Leaf River Basin and is intended primarily for pedagogical and demonstration purposes. Users interested in reproducing or extending the analyses to the CAMELS-US catchments are encouraged to follow the methodology described in Wang & Gupta (2025) and use the scripts provided in the original [Zenodo](https://doi.org/10.5281/zenodo.15811866).

## Quick Start

### Requirements

Use Python 3.10 and install the dependencies listed in `requirements.txt`, including NumPy, pandas, PyTorch, and scikit-learn. Run the following command from the repository root:

```bash
python -m pip install -r requirements.txt
```

### Leaf River Data

The example data are provided in `20220527-MDUPLEX-LeafRiver/`, the default data directory used by the training and evaluation scripts.

- `LeafRiverDaily_43YR.txt`: daily precipitation (`P`), potential evapotranspiration (`PET`), and streamflow (`Q`), in that column order and all in mm/day.
- `LeafRiverDaily_43YR_Flag.txt`: flags identifying the data subsets used for model spin-up, training, selection, and testing.

| Flag | Data subset |
| ---: | :--- |
| `-99999` | Spin-up |
| `-1` | Training |
| `0` | Selection |
| `1` | Testing |

The data sequence includes three repetitions of water year 1949 for spin-up, followed by the 40-year simulation period covering water years 1949–1988.

### Pretrained Checkpoints

Pretrained Leaf River checkpoints are included in the model folders. They can be used for direct evaluation or as starting points for continued training and fine-tuning.

## Single-Layer MCP Networks

The single-layer experiments investigate a family of parallel MCP networks in which the number of MCP nodes, the treatment of network-level mass conservation, and the degree of internal state information sharing can be systematically varied.

Figure 2 summarizes the five network-level architectural hypotheses considered in Wang & Gupta (2025).

![Single-layer MCP network architectures](Figure2.png)

*Figure 2. Single-layer MCP network architectures considered in this study. The Distributed-Input (DI) and Distributed-State (DS) formulations conserve mass at the overall network level, whereas DIR, DSR, and MLB progressively relax the network-level mass-conservation constraints.*

### Network Types

Five network types are implemented:

| Network type | Paper notation | Input distribution | Output aggregation |
| --- | --- | --- | --- |
| Distributed-Input | MN<sup>DI</sup>(N<sub>1</sub>) | Positive weights constrained to sum to 1 | Sum of all nodal outputs |
| Distributed-State | MN<sup>DS</sup>(N<sub>1</sub>) | Unity input to every node | Positive weights constrained to sum to 1 |
| Distributed-Input-Relaxed | MN<sup>DIR</sup>(N<sub>1</sub>) | Positive weights not constrained to sum to 1 | Sum of all nodal outputs |
| Distributed-State-Relaxed | MN<sup>DSR</sup>(N<sub>1</sub>) | Unity input to every node | Positive weights not constrained to sum to 1 |
| Machine-Learning Benchmark | MN<sup>MLB</sup>(N<sub>1</sub>) | Unconstrained linear mapping with bias | Unconstrained linear mapping with bias |

In the **Distributed-Input (DI)** formulation, precipitation is partitioned among the parallel MCP nodes using positive weights that sum to one, while the resulting nodal streamflow components are summed to obtain total streamflow. In contrast, the **Distributed-State (DS)** formulation sends the full precipitation input to every node and combines the resulting streamflow components using positive output weights that sum to one.

The **DIR** and **DSR** formulations relax the corresponding input- and output-side network-level mass-conservation constraints, respectively. The **MLB** architecture provides the least constrained network-level benchmark by using unconstrained linear mappings with bias terms at both the input and output sides.

### Number of Nodes

The single-layer networks are denoted as:

$$
MN^{\mathrm{Type}}_{\mathrm{Sharing}}(N_1)
$$

where $N_1$ denotes the number of MCP nodes in the first hidden layer. Because the architectures considered in this section contain only one hidden layer, $N_1$ is also the total number of parallel MCP nodes in the network.

In the experiments reported in Wang & Gupta (2025), we evaluated:

$$
N_1 = 2,\ 3,\ 4,\ 5
$$

For example,

$$
MN^{DS}_{None}(3)
$$

denotes a single-layer **Distributed-State** network with **three MCP nodes** and **no internal state information sharing**.

### Internal State Information Sharing

For each of the five network types, four information-sharing configurations are considered following the notation used in Table 1 of Wang & Gupta (2025):

| Sharing option | Paper notation | Information available to each node |
| --- | --- | --- |
| None | `None` | No state sharing |
| Sharing-Augmented Output | `SAO` | States shared with the output gate only |
| Sharing-Augmented Loss | `SAL` | States shared with the loss gate only |
| Sharing-Augmented Loss and Output | `SALO` | States shared with both output and loss gates |

Combining the five network types with the four information-sharing configurations gives **20 single-layer architectural variants** for each value of $N_1$.

Examples include:

- $MN^{DS}_{None}(N_1)$: Distributed-State with no information sharing
- $MN^{DS}_{SAO}(N_1)$: Distributed-State with state information shared at the output gates
- $MN^{DS}_{SAL}(N_1)$: Distributed-State with state information shared at the loss gates
- $MN^{DS}_{SALO}(N_1)$: Distributed-State with state information shared at both the output and loss gates

The same `None`, `SAO`, `SAL`, and `SALO` convention is used for the DI, DIR, DSR, and MLB network families.

### Single-Layer Model Space

The complete single-layer model space consists of:

- 5 network types: `DI`, `DS`, `DIR`, `DSR`, and `MLB`
- 4 information-sharing options: `None`, `SAO`, `SAL`, and `SALO`
- 4 network sizes: $N_1 = 2, 3, 4, 5$

This gives:

$$
5 \times 4 \times 4 = 80
$$

single-layer model configurations.

### Running a Single-Layer Case

Each model variant is provided with a training script and a corresponding evaluation script. The filenames identify both the network type and the information-sharing configuration.

For example, to run the **Machine-Learning Benchmark (MLB)** network without internal state sharing:

```bash
python train_MLB_None_single_layer.py

## References & Citation

Readers should consult the original paper for the formal model names, notation, and mathematical definitions associated with each MCP variant.

- Full article: [Wang & Gupta (2025)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025WR040178)
- DOI: [10.1029/2025WR040178](https://doi.org/10.1029/2025WR040178)

> Wang, Y.-H., & Gupta, H. V. (2025). Using machine learning to discover parsimonious and physically‐interpretable representations of catchment‐scale rainfall‐runoff dynamics. *Water Resources Research, 61*(12), e2025WR040178. https://doi.org/10.1029/2025WR040178

## Contact

This repository provides a consolidated and reproducible collection of MCP neural network variants, including cleaned training and evaluation scripts, checkpoints, and data conventions. 

For questions or suggestions, please contact **Yuan-Heng Wang, Ph.D.** at `yhwang0730@gmail.com` or `yhwang0730@arizona.edu`.

## License

This project is licensed under the [MIT License](LICENSE).
