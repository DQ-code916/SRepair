# Learning Dependency Models for Subset Repair

## File Structure

+ algorithm: source code of Probabilistic, Clique and ILP  
+ data: some source files of datasets used in experiments  
+ util: auxiliary functions and classes  
+ Appendix.pdf: proofs for all the theoretical results in the manuscript, including Theorem 1, Proposition 2 to 16  
+ main.py: source code of the experiments
+ main_core.py: source code of the whole process, including data loading, conflict detection, conformance calculation and the main process of the three algorithms

## Script Running  

```
# change the data path in main_core.py
# change the tasks for different experiments in main.py
# run the script
python main.py
```


```
python main.py --============ restaurant ============ --1-complete  --2-complete --Detection time: 1.319 --3-complete --Handling time:9.012 --Probabilistic --Time:0.009  --Clique --Model solved successfully  --Time:0.12 --  Pre  Rec  F1  Time  --Probabilistic  0.895  0.839  0.866  0.005  --Clique  0.903  0.868  0.885  0.12
```




## Datasets:
+ Flights, Rayyan:  https://github.com/BigDaMa/raha
+ Restaurant: https://github.com/densitysrepair/densitysrepair
+ Soccer:  https://db.unibas.it/projects/bart/
+ SPStock:  https://github.com/RangerShaw/FastADC
+ Iris, Yeast: https://sci2s.ugr.es/keel/attributeNoise.php

## Tools:
+ Bart: a tool for error generation. https://db.unibas.it/projects/bart/
+ FastADC: a tool for DC discovery. https://github.com/RangerShaw/FastADC