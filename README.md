# Mechanistic Modeling of pH Transients in Affinity Chromatography Incorporating Cation and Anion Exchange Interactions and Multivalent Ion Retention

## Overview

This repository contains the Python code and inlet concentration profiles used to simulate pH transients during buffer transitions in Protein A affinity chromatography, as described in:

A.R. Moser, S.H. Altern, S. Bhoyar, R.H. Zha, A.M. Lenhoff, S.M. Cramer, Mechanistic Modeling of pH Transients in Affinity Chromatography Incorporating Cation and Anion Exchange Interactions and Multivalent Ion Retention, manuscript in preparation.

The mechanistic model accounts for convective transport, acid–base equilibria, and interactions between mobile-phase ions and strong and weak cation and anion exchange ligand groups. Six scripts are provided for the different buffer systems investigated in the study.

## Repository Structure

```text
.
├── tris_acetate.py
├── tris_glycine.py
├── tris_citrate.py
├── phosphate_acetate.py
├── phosphate_glycine.py
├── phosphate_citrate.py
├── input_profiles/       # Inlet concentration profiles (.csv)
├── example_output/       # Example simulation results
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
├── LICENSE               # MIT License
└── .gitignore            # Excluded generated files
```

## Requirements

The simulations were performed using Python 3.12.10. The required packages and their versions are listed in `requirements.txt` and are the same for all six scripts.

Install the dependencies using:

```bash
python -m pip install -r requirements.txt
```

## Running the Simulations

Each script simulates a different buffer system and can be run independently. For example:

```bash
python tris_acetate.py
```

The inlet concentration profiles are provided as CSV files in the `input_profiles/` directory. The files contain time-dependent concentrations of the buffer and salt species entering the column.

The experiments to simulate are selected by modifying the `filenames` list near the bottom of each script. Model parameters, including column properties, operating conditions, ligand densities, and dissociation parameters, are specified near the beginning of each script.

The simulations generate CSV files containing the predicted column outlet pH, ionic strength, and species concentrations, along with plots of the results. Example outputs are provided in `example_output/`.

## Citation

If you use this code, please cite the accompanying research article:

A.R. Moser, S.H. Altern, S. Bhoyar, R.H. Zha, A.M. Lenhoff, S.M. Cramer, Mechanistic Modeling of pH Transients in Affinity Chromatography Incorporating Cation and Anion Exchange Interactions and Multivalent Ion Retention, manuscript in preparation.

The code is permanently archived on Zenodo:

https://doi.org/10.5281/zenodo.XXXXXXX

## License

This software is distributed under the MIT License. See the LICENSE file for details.
