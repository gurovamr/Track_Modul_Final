# first_blood
Extension of the FirstBlood 1D–0D hybrid hemodynamic solver with a modular pipeline for integrating patient-specific Circle of Willis (CoW) geometry into a full systemic circulation model.

This repository combines:
- The original FirstBlood solver
- A data-generation pipeline for injecting patient-specific cerebral geometry
- Automated numerical and biological validation scripts

### Original FirstBlood Solver
1D and 0D combined hemodynamic simulator for the human circulatory system utilizing the Method of Characteristics and MacCormack scheme.

Originally developed for research and education at:
Budapest University of Technology and Economics
Faculty of Mechanical Engineering
Department of Hydrodynamic Systems

#### Original Authors
Dr. Richárd Wéber – Assistant Professor
Márta Viharos – BSc student
Dániel Gyürki – Research Assistant Fellow

Wéber, R., Gyürki, D., & Paál, G. (2023).
First blood: An efficient, hybrid one- and zero-dimensional, modular hemodynamic solver.
International Journal for Numerical Methods in Biomedical Engineering.
DOI: 10.1002/cnm.3701

## This Work: Patient-Specific CoW Pipeline
This project extends the FirstBlood framework by:
- Injecting patient-specific Circle of Willis geometry (radius and length)
- Mapping patient JSON data to the Abel_ref2 systemic topology
- Automating model generation
- Performing numerical stability validation
- Computing biological hemodynamic metrics
- Visualizing cerebral flow distribution
- The systemic topology remains based on the Abel_ref2 reference model, while cerebral vessel geometry is updated per patient.

## Project Strtucture
- source/              → Core FirstBlood solver
- projects/            → Compilable simulations
- models/              → Model input CSV files
- pipeline/            → Patient data integration & analysis scripts 

### Build & Run
make -f make_*.mk

#### Dependencies 
- C++ compiler (clang++ recommended)
- Eigen
- make
- Python 3 (numpy, matplotlib, pandas)

