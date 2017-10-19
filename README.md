A Nearest Neighbor Classifier for High-Speed Big Data Streams with Instance Selection (spark-IS-streaming)
==========================================================

Here we present an efficient nearest neighbor solution to classify fast and massive data streams using Apache Spark. 
It is formed by a distributed case-base and an instance selection method that enhances its performance and effectiveness. 
A distributed metric tree (based on M-trees) has been designed to organize the case-base and consequently to speed up the
neighbor searches. This distributed tree consists of a top-tree (in the master node) that routes the searches in the first levels
and several leaf nodes (in the slaves nodes) that solve the searches in next levels through a completely parallel scheme.

A improved local version of RNGE [1] in order to control the insertion and removal of noisy instances.
For each incoming example, a relative graph is built around each new instance and its subset of neighbors. 
The local graphs are then used to edit the case-base by deciding what instances should be inserted, removed or left intact.

Associated Spark package: https://spark-packages.org/package/sramirez/spark-IS-streaming

Associated journal paper: S. Ramírez-Gallego, B. Krawczyk, S. García, M. Woźniak, J. M. Benítez and F. Herrera, "Nearest Neighbor Classification for High-Speed Big Data Streams Using Spark," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 47, no. 10, pp. 2727-2739, Oct. 2017.
doi: 10.1109/TSMC.2017.2700889
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7993020&isnumber=8038140


## Parameters:

Our distributed approach includes several user-defined input parameters, which are described below:

* "type": type of file. "keel" for KEEL/ARFF files, "labeled" for Spark generated files with labeled points, or "csv" for standard csv files.
Default: "csv".
* "header": path of header file (only valid for KEEL files). 
* "output": path of output directory.
* "npart": number of partitions for the first partitioning process. Default: 4.
* "k": number of neighbors selected for prediction. Default: 1.
* "kGraph": number of neighors selected to construct local graphs. Default: 10.
* "rate": number of examples in each batch. Default: 2500.
* "interval": the time interval at which streaming data will be divided into batches. Default: 1000 (ms).
* "seed": seed value for random generator. Default: 237597430.
* "ntrees": number of sub-trees in the slave nodes. It should be set equal or higher than the number of initial partitions. Default: 4.
* "overlap": the distance between elements in sub-trees. 0 means sub-trees are disjoint. Default: 0.
* "edited": if filtering of new noisy examples ocurrs. Default: false.
* "removedOld": if removal of already inserted examples considered as noise ocurrs. Default: false.
* "timeout": milleseconds before shutting the streaming execution. Default: 600000.
* "sampling": percent of sampling w/o replacement on original static data. Default: 0.0.
* "nClass": number of classes in data. Default: 2 (binary).

## Example: 

spark-submit --class org.ugr.sci2s.mllib.test.QueuRDDStreamingTest spark-IS-streaming.jar 
--input=hdfs://localhost:8020/train.data --output=hdfs://localhost:8020/output/streaming-test 
--type=csv --interval=1000 --rate=100000 --ntrees=460 --npart=460 --edited=true

For a more thorough sourc code example, please refer to: 
src/main/scala/org/ugr/sci2s/mllib/test/QueuRDDStreamingTest.scala
  
## Contributors

- Sergio Ramírez-Gallego (sramirez@decsai.ugr.es) (main contributor and maintainer).

## References

[1] J.S. Sánchez, F. Pla, F.J. Ferri, Prototype selection for the nearest neighbour rule
through proximity graphs, Pattern Recognition Lett. 18 (1997) 507–513.
KEEL project: https://github.com/SCI2SUGR/KEEL / http://sci2s.ugr.es/keel/description.php
