---

## Physical-Conceptual MA3–MA6 Models with State Sharing

The `MA-Sharing-SALO/` directory contains information-sharing extensions of four physical-conceptual MCP architectures — **MA3, MA4, MA5, and MA6**.

These architectures were originally introduced and evaluated in our previous study:

> Wang, Y.-H., & Gupta, H. V. (2024). *Towards interpretable physical-conceptual catchment-scale hydrological modeling using the mass-conserving-perceptron*. Water Resources Research, 60(10), e2024WR037224. https://doi.org/10.1029/2024WR037224

In that earlier work, MA3–MA6 were developed as progressively more complex architectural hypotheses for representing catchment-scale rainfall-runoff dynamics using physically interpretable and mass-conserving MCP units.

Wang & Gupta (2025) subsequently revisited these architectures as physical-conceptual benchmarks when investigating whether model behavior can benefit from allowing information about the overall distribution of internal cell states to influence individual gating functions.

### Base MA3–MA6 Architectures

The four architectures differ in the number of cell states and in the organization of the associated runoff pathways:

| Model | Main architectural hypothesis |
| --- | --- |
| `MA3` | One flow pathway with two cell states in series: a catchment soil-moisture storage followed by a routing storage |
| `MA4` | Two parallel flow pathways representing near-surface runoff and groundwater/baseflow |
| `MA5` | Three cell states and two flow pathways, extending MA4 by including explicit surface/channel routing |
| `MA6` | Extends MA5 by including an additional overland-flow pathway that can route water toward the catchment outlet |

All four architectures retain the physically interpretable storage and flow-path structures developed in Wang & Gupta (2024).

### SALO State Sharing

The implementations provided in this repository augment the original MA3–MA6 architectures using **Sharing-Augmented Loss and Output (SALO)** gates.

In the original physical-conceptual formulations, the gating behavior associated with a given MCP state is primarily determined using information local to that state.

In the SALO extensions, both the **output gate** and the **loss gate** are additionally allowed to use information from other cell states in the model.

Conceptually:

```text
Original MA model:

state i
  ├── output gate <- local state information
  └── loss gate   <- local state information


MA-SALO model:

state i
  ├── output gate <- local state + information from other cell states
  └── loss gate   <- local state + information from other cell states
```

The underlying MA3–MA6 storage and flow-path architectures are retained. The modification concerns the information available to the gating functions rather than the introduction of new hydrologic states or new flow pathways.

This setup provides a direct way to investigate whether fluxes associated with an individual conceptual storage should depend only on its local state or on a broader representation of the moisture distribution across the entire model.

### Available MA-SALO Cases

Four information-sharing physical-conceptual models are included:

| Repository folder | Base architecture | Information sharing |
| --- | --- | --- |
| `MA3-SALO` | MA3 | Output + loss gates |
| `MA4-SALO` | MA4 | Output + loss gates |
| `MA5-SALO` | MA5 | Output + loss gates |
| `MA6-SALO` | MA6 | Output + loss gates |

The directory is organized as:

```text
MA-Sharing-SALO/
├── MA3-SALO/
├── MA4-SALO/
├── MA5-SALO/
└── MA6-SALO/
```

Each case contains a pretrained Leaf River checkpoint together with the corresponding training and evaluation scripts.

For example:

```text
MA-Sharing-SALO/
└── MA6-SALO/
    ├── model_epoch8.pt
    ├── train_MA6_SALO.py
    └── evaluate_MA6_SALO.py
```

The exact checkpoint filename may differ among cases because each architecture retains its corresponding pretrained model.

### Running an MA-SALO Case

For example, to evaluate the pretrained **MA6-SALO** model:

```bash
cd MA-Sharing-SALO/MA6-SALO
python evaluate_MA6_SALO.py
```

To continue training from the checkpoint stored in the same directory:

```bash
python train_MA6_SALO.py
```

The same naming convention is used for the remaining architectures:

```text
train_MA3_SALO.py
evaluate_MA3_SALO.py

train_MA4_SALO.py
evaluate_MA4_SALO.py

train_MA5_SALO.py
evaluate_MA5_SALO.py

train_MA6_SALO.py
evaluate_MA6_SALO.py
```

By default, each script uses the pretrained checkpoint stored in its corresponding model directory.

The evaluation scripts report model skill for the spin-up, training, selection, and testing periods and save the associated discharge, storage, routing, loss, and gate diagnostics.

---

## References & Citation

Readers should consult the original paper for the formal model names, notation, and mathematical definitions associated with each MCP variant.

- Full article: [Wang & Gupta (2025)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025WR040178)
- DOI: [10.1029/2025WR040178](https://doi.org/10.1029/2025WR040178)

> Wang, Y.-H., & Gupta, H. V. (2025). Using machine learning to discover parsimonious and physically-interpretable representations of catchment-scale rainfall-runoff dynamics. *Water Resources Research, 61*(12), e2025WR040178. https://doi.org/10.1029/2025WR040178

---

## Contact

This repository provides a consolidated and reproducible collection of MCP neural network variants, including cleaned training and evaluation scripts, checkpoints, and data conventions.

For questions or suggestions, please contact **Yuan-Heng Wang, Ph.D.** at `yhwang0730@gmail.com` or `yhwang0730@arizona.edu`.

---

## License

This project is licensed under the [MIT License](LICENSE).
