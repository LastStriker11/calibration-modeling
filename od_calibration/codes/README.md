﻿**Update date:** *15/11/2019*

1. Integrate the scenario_generator into the main program.
---------------------------
**Update date:** *12/11/2019*

1. Add a variable in `Network_var` called `objective` to indicate the objective of the calibration (traffic counts calibration or travel time calibration).
   
----------------------------
**Update date:** *25/10/2019*

1. Folder **_parallel_** contains the re-organized code for parallel processing. There are four files in this folder:
   - `Basic_scripts.py`: has been removed from the latest version.
   - `Functions.py`: functions to be imported into the `main`. Please take care of the attention text written at the very beginning. In order to make it become general, you will see there are four functions with the suffix **PC_SPSA** or **SPSA**, i.e. `generation_PC_SPSA`, `generation_SPSA`, `rep_generation_PC_SPSA`, and `rep_generation_SPSA`. In fact, `generation_PC_SPSA` and `generation_SPSA` are used to realize the same function, while `rep_generation_PC_SPSA` and `rep_generation_SPSA` are used to realize another function. As a result, this file `Functions.py` can be applied to both of PC_SPSA algorithm and SPSA algorithm (of course `Basic_scripts.py` is the same always). The features of different functions are commented above of the functions. Hope they are clear.
   - `main_PC_SPSA_parallel.py` and `main_SPSA_parallel.py` are the main sections  for PC_SPSA and SPSA, respectively.

2. Folder **_t_test_NSumoRep_** contains the code for testing the necessary number of replication of SUMO simulation to obtain statistically significant results. In which, `t_test_NSumoRep.py` is used to run SUMO simulations 100 times and collect the simulation results, i.e. traffic counts and travel time. While `t_test_analysis.py` is used to import the result from `t_test_NSumoRep.py`  and  test the necessary replications for traffic counts and travel time.







