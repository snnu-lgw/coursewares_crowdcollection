# Type diversity maximization aware coursewares crowdcollection in MOOCs

The following describes the code of the experiment.

$Distribution\_Selection$, $Number\_Task$, $Number\_Type$, $Number\_Worker$, $Beta$, $MaxGeneration$ and $Budget$ are the system input parameters.

$Distribution_Selection$ = {$U$, $N$, $E$ | $U$, $N$, $E$ are uniform distribution, normal distribution and exponential distribution.}

$Number_Task$ = ${1000, 2000, 3000, 4000, 5000}$

$Number_Type$ = ${500, 1000, 1500, 2000, 2500}$

$Number_Worker$ = ${2000, 3000, 4000, 5000, 6000}$

$Beta$ = ${0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5}$

$MaxGeneration$ = $300$

$Budget = ${80, 170, 350}$

# How to run

### Generate simulation data submitted by users:
```
python Data_Generator.py Distribution_Selection Number_Task Number_Type Number_Worker
```
### Generate the data needed by genetic algorithm:
```
python GA-Data-Generator.py Distribution_Selection Number_Task Number_Type Number_Worker Budget
```
### Run the code
```
python CC-Eba.py Distribution_Selection Number_Task Number_Type Number_Worker
python CC-Wba.py Distribution_Selection Number_Task Number_Type Number_Worker
python GA-CC-Eba.py Distribution_Selection Number_Task Number_Type Number_Worker MaxGeneration Budget
python GA-CC-Wba.py Distribution_Selection Number_Task Number_Type Number_Worker MaxGeneration Beta Budget
```
