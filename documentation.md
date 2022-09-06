## Custom Calibration and Modeling Toolkit

This toolbox is used to calibrate large-scale [SUMO](https://sumo.dlr.de/docs/index.html) networks 
using state-of-the-art calibration algorithms. This toolbox is updated constantly 
to include more algorithms, and allow more flexible simulation and network setups.

Currently, this toolbox only includes the [PC-SPSA](https://www.sciencedirect.com/science/article/pii/S0968090X21004903) algorithm.

Please refer to the documentation of respective modules for more details:

 - [preparation.py](./preparation.html)
 - [calibration_algorithms.py](./calibration_algorithms.html)
 - [sumo_operation.py](./sumo_operation.html)
 - [evaluation_metrics.py](evaluation_metrics.html)

This toolbox is developed by [Chair of Transportation Systems (TSE)](https://www.mos.ed.tum.de/en/vvs/home/) at [Technical University of Munich (TUM)](https://www.tum.de/en/). 
If you use this toolbox/platform in your workm please cite the following publication:

     @article{qurashi2022dynamic,
       title={Dynamic demand estimation on large scale networks using Principal Component Analysis: The case of non-existent or irrelevant historical estimates},
       author={Qurashi, Moeid and Lu, Qing-Long and Cantelmo, Guido and Antoniou, Constantinos},
       journal={Transportation Research Part C: Emerging Technologies},
       volume={136},
       pages={103504},
       year={2022},
       publisher={Elsevier}
     }