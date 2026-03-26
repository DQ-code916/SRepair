# SRepair

## Truth $\neq$ Frequency: Leveraging Dependencies for Subset Repair

  

## Setup Instructions

  

### Prerequisites

- Python 3.7 or higher

- pip (Python package manager)

  

### Installation

  

1. **Navigate to the project directory:**

```bash

cd <project_directory>

```

  

2. **Create a virtual environment:**

```bash

# On Windows

python -m venv repair_env

repair_env\Scripts\activate

# On macOS/Linux

python3 -m venv repair_env

source repair_env/bin/activate

```

  

3. **Install dependencies:**

```bash

pip install -r requirements.txt

```

  



  

### Important Notes

  

> **Gurobi Solver License:** This project uses Gurobi solver (gurobipy) for Clique.

> - You need a valid Gurobi license to run Clique 

> - Free academic licenses are available at: https://www.gurobi.com/academia/academic-program-and-licenses/

> - Install Gurobi license file after installing the package

  



  

## File Structure

  

+ `algorithm/`: source code of Probabilistic, Clique and ILP algorithms

+ `data/`: source files of datasets

+ `util/`: auxiliary functions and classes

+ `experiments/`: experimental code and table embedding evaluation

- `main.py`: source code for running experiments



- `table_embedding/`: table embedding similarity calculation scripts

+ `main_core.py`: core implementation including data loading, conflict detection, conformance calculation and the main process of the three algorithms

+ `run.py`: script for running the main algorithms

+ `appendix.pdf`: appendix with proofs for all theoretical results (Theorem 1, Proposition 2 to 17), examples and some extra experiments 

+ `requirements.txt`: Python package dependencies

  

## Script Running

  

### Running Main Algorithms

  

To run the main algorithms (Probabilistic, Clique, ILP):

  

```bash

python run.py

```

  

### Running Experiments

  

To run experimental code:

  

```bash

python experiments/main.py

```

  

Before running, please configure the dataset paths in the respective script files.

  

### Expected Output

  

The scripts output Precision, Recall, F1-score, and execution time for each algorithm, along with detailed progress messages showing different stages of execution.

  

#### Example Output

  

```

============ restaurant ============

1-complete

2-complete

Detection time: 1.319

3-complete

Handling time: 9.012

==========================================

---------- Probabilistic ----------

Time: 0.009

p= 0.895

r= 0.839

f1= 0.866

==========================================

---------- Clique ----------

Model solved successfully

Time: 0.12

p= 0.903

r= 0.868

f1= 0.885

```

  
  
  
  

## Datasets

  

+ **Flights, Rayyan**: https://github.com/BigDaMa/raha

+ **Company**: https://www.kaggle.com/datasets/jacksapper/company-sentiment-by-location

+ **Restaurant**: https://github.com/densitysrepair/densitysrepair

+ **Soccer**: https://db.unibas.it/projects/bart/


+ **Inspection**:https://github.com/RangerShaw/FastADC

+ **Income**: https://github.com/socialfoundations/folktables

+ **Parking (NYC Parking Tickets)**: https://www.kaggle.com/datasets/new-york-city/nyc-parking-tickets?resource=download

+ **Iris, Yeast**: https://sci2s.ugr.es/keel/attributeNoise.php

+ **AirQuality**: https://github.com/marisuki/LearnCRR

+ **SocialMedia**: https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity




  

## Tools

  

+ **Gurobi**: Optimization solver used for Clique and ILP

- Website: https://www.gurobi.com/

- Academic licenses: https://www.gurobi.com/academia/academic-program-and-licenses/


+ **Bart**: Tool for error generation. https://db.unibas.it/projects/bart/

+ **FastADC**: Tool for DC discovery. https://github.com/RangerShaw/FastADC
