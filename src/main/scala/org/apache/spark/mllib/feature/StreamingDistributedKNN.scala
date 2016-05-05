/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feature

import scala.reflect.ClassTag
import scala.util.hashing.byteswap64
import scala.collection.mutable.Queue
import collection.JavaConversions._
import breeze.linalg.{DenseVector => BDV, Vector => BV}
import breeze.stats._
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.SparkContext._
import org.apache.spark.Logging
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaSparkContext._
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.api.java.{JavaDStream, JavaPairDStream}
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.Partitioner
import org.apache.spark.rdd.ShuffledRDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.knn._
import mtree.DataLP
import org.apache.spark.mllib.feature.StreamingDistributedKNN._
import org.apache.spark.util.BoundedPriorityQueue
import mtree.IndexedLP

/**
 * StreamingKNNModel extends MLlib's KMeansModel for streaming
 * algorithms, so it can keep track of a continuously updated weight
 * associated with each cluster, and also update the model by
 * doing a single iteration of the standard k-means algorithm.
 *
 * The update algorithm uses the "mini-batch" KMeans rule,
 * generalized to incorporate forgetfullness (i.e. decay).
 * The update rule (for each cluster) is:
 *
 * {{{
 * c_t+1 = [(c_t * n_t * a) + (x_t * m_t)] / [n_t + m_t]
 * n_t+t = n_t * a + m_t
 * }}}
 *
 * Where c_t is the previously estimated centroid for that cluster,
 * n_t is the number of points assigned to it thus far, x_t is the centroid
 * estimated on the current batch, and m_t is the number of points assigned
 * to that centroid in the current batch.
 *
 * The decay factor 'a' scales the contribution of the clusters as estimated thus far,
 * by applying a as a discount weighting on the current point when evaluating
 * new incoming data. If a=1, all batches are weighted equally. If a=0, new centroids
 * are determined entirely by recent data. Lower values correspond to
 * more forgetting.
 *
 * Decay can optionally be specified by a half life and associated
 * time unit. The time unit can either be a batch of data or a single
 * data point. Considering data arrived at time t, the half life h is defined
 * such that at time t + h the discount applied to the data from t is 0.5.
 * The definition remains the same whether the time unit is given
 * as batches or points.
 */
class StreamingDistributedKNNModel (
    val topTree: Broadcast[MTreeWrapper],
    val trees: RDD[MTreeWrapper],
    var tau: Double) extends Serializable with Logging {
  
  private def topByKey(toOrder: RDD[(Long, (DataLP, Double, Double))], num: Int)
    (implicit ord: Ordering[(DataLP, Double, Double)]) = {
    toOrder.aggregateByKey(new BoundedPriorityQueue[(DataLP, Double, Double)](num)(ord))(
      seqOp = (queue, item) => {
        queue += item
      },
      combOp = (queue1, queue2) => {
        queue1 ++= queue2
      }
    ).mapValues(_.toArray.sorted(ord.reverse))  // This is an min-heap, so we reverse the order.
  }
  
  private[feature] def accuratekNNQuery(data: RDD[LabeledPoint], k: Int): RDD[(TreeLP, Array[TreeLP])] = {

    // Efficient join if elements and their neighbors have the same partitioning scheme
    val indexedData = data.zipWithIndex.map(_.swap).partitionBy(new HashPartitioner(trees.partitions.length)).cache
    val searchData = indexedData.flatMap { case (idx, vector) =>
        val indices = searchIndices(vector.features, topTree.value, tau)
          .map(i => (i, (vector, idx)))
        assert(indices.filter(_._1 == -1).length == 0, s"indices must be non-empty: $vector")
        indices
    }.partitionBy(new HashPartitioner(trees.partitions.length))

    // for each partition, search points within corresponding child tree
    val results = searchData.zipPartitions(trees, preservesPartitioning = true) {
      (childData, trees) =>
        val tree = trees.next()
        assert(!trees.hasNext)
        childData.flatMap {
          case (ipart, (point, idx)) =>
            tree.kNNQuery(k, point.features.toArray.map(_.toFloat))
              .map(pair => (idx, (pair.data.asInstanceOf[DataLP], pair.distance, ipart.toDouble)))
        }
    }
    
    // merge results by point index together and keep topK results
    val neigs = topByKey(results, k)(Ordering.by(_._2)).map { case (lidx, iter) => 
      lidx -> iter.map{ case(neigh, _, idx) => new TreeLP(LPUtils.fromJavaLP(neigh.asInstanceOf[DataLP]), idx.toInt, null)}
    }
    
    neigs.join(indexedData).values.map{ case(neigs, lp) =>
      val tlp = new TreeLP(lp, searchIndex(lp.features, topTree.value), null)
      tlp -> neigs
    }
  }
  
  private def accuratekNNPrediction(data: RDD[LabeledPoint], k: Int): RDD[(Double, Double)] = {

    // Efficient join if elements and their neighbors have the same partitioning scheme
    val searchData = data.zipWithIndex.map(_.swap).flatMap { case (idx, vector) =>
        val indices = searchIndices(vector.features, topTree.value, tau)
          .map(i => (i, (vector, idx)))
        assert(indices.filter(_._1 == -1).length == 0, s"indices must be non-empty: $vector")
        indices
    }.partitionBy(new HashPartitioner(trees.partitions.length))

    // for each partition, search points within corresponding child tree
    val results = searchData.zipPartitions(trees, preservesPartitioning = true) {
      (childData, trees) =>
        val tree = trees.next()
        assert(!trees.hasNext)
        childData.flatMap {
          case (ipart, (point, idx)) =>
            tree.kNNQuery(k, point.features.toArray.map(_.toFloat))
              .map(pair => (idx, (pair.data.asInstanceOf[DataLP], pair.distance, point.label)))
        }
    }
    
    // merge results by point index together and keep topK results
    topByKey(results, k)(Ordering.by(_._2)).map { case (lidx, neig) => 
      val pred = neig.map(_._1.getLabel).groupBy(identity).maxBy(_._2.size)._1
      neig(0)._3 -> pred
    }
  }
  
  private def fastkNNQuery(data: RDD[LabeledPoint], k: Int): RDD[(TreeLP, Array[TreeLP])] = {

    val searchData = data.map { vector =>
        val indices = searchIndex(vector.features, topTree.value)
        assert(indices != -1, s"indices must be non-empty: $vector")
        indices -> vector
    }.partitionBy(new HashPartitioner(trees.partitions.length))

    // for each partition, search points within corresponding child tree
    searchData.zipPartitions(trees) {
      (childData, trees) =>
        val tree = trees.next()
        assert(!trees.hasNext)
        childData.map {
          case (ipart, point) =>
            val lp = new TreeLP(point, ipart, null)
            lp -> tree.kNNQuery(k, point.features.toArray.map(_.toFloat))
              .map(pair => new TreeLP(LPUtils.fromJavaLP(pair.data.asInstanceOf[DataLP]), ipart, null))
              .toArray            
        }  
    }
    
  }
  
  def fastEdition(data: RDD[LabeledPoint], kGraph: Int, removeOld: Boolean): RDD[MTreeWrapper] = {
    
    val searchData = data.map { vector =>
        val indices = searchIndex(vector.features, topTree.value)
        assert(indices != -1, s"indices must be non-empty: $vector")
        indices -> vector
    }.partitionBy(new HashPartitioner(trees.partitions.length))

    // for each partition, search and edit points within corresponding child tree
    searchData.zipPartitions(trees) {
      (childData, trees) =>
        val tree = trees.next()
        assert(!trees.hasNext)
        childData.foreach {
          case (ipart, point) =>
            val lp = new TreeLP(point, ipart, null)
            val neighbors = tree.kNNQuery(kGraph, point.features.toArray.map(_.toFloat))
              .map(pair => new TreeLP(LPUtils.fromJavaLP(pair.data.asInstanceOf[DataLP]), ipart, null))
              .toArray   
            val edited = InstanceSelection.localInstanceSelection(lp -> neighbors, removeOld)
            edited.foreach{ lp => 
              if(lp.action == INSERT)
                tree.insert(LPUtils.toJavaLP(lp.point))
              else if (lp.action == REMOVE)
                tree.remove(LPUtils.toJavaLP(lp.point))
            }            
        } 
        Iterator(tree)
    }    
  }
  
  def predict(data: RDD[LabeledPoint], k: Int): RDD[(Double, Double)] = {
    if(topTree != null && !data.isEmpty()) {
      if(tau > 0) {
        accuratekNNPrediction(data, k)
      } else {
        fastkNNQuery(data, k).map{ case (lp, arr) =>
          val pred = arr.map(_.point.label).groupBy(identity).maxBy(_._2.size)._1
          (lp.point.label, pred)
        }
      }
         
    } else {
      data.context.emptyRDD[(Double, Double)]
    }
  } 
  
}

/**
 * StreamingKNN provides methods for configuring a
 * streaming k-means analysis, training the model on streaming,
 * and using the model to make predictions on streaming data.
 * See KMeansModel for details on algorithm and update rules.
 *
 * Use a builder pattern to construct a streaming k-means analysis
 * in an application, like:
 *
 * {{{
 *  val model = new StreamingKNN()
 *    .setDecayFactor(0.5)
 *    .setK(3)
 *    .setRandomCenters(5, 100.0)
 *    .trainOn(DStream)
 * }}}
 */
class StreamingDistributedKNN (    
    var kGraph: Int,
    var nTrees: Int,
    var overlapDistance: Double,
    var sampleSizes: Array[Int],
    var edited: Boolean,
    var removeOld: Boolean,
    var seed: Long,
    var initialRDD: RDD[LabeledPoint]) extends Logging with Serializable {

  def this() = this(10, 2, -1, (100 to 1000 by 100).toArray, true, false, 26827651492L, null)
  
  private val DEFAULT_NUMBER_SAMPLES = 10
  
  protected var model: StreamingDistributedKNNModel = {
    if(initialRDD == null) {
      new StreamingDistributedKNNModel(null, null, 0.0)
    } else {
      initializeModel(initialRDD)
    }      
  } 
    

  /**
   * Set the number of partitions/sub-trees to use in the model.
   */
  def setKGraph(k: Int): this.type = {
    require(k > 1)
    this.kGraph = k
    this
  }
  
  def setEdited(edited: Boolean): this.type = {
    this.edited = edited
    this
  }
    
  def setRemovedOld(ro: Boolean): this.type = {
    this.removeOld = ro
    this
  }    
  
  def setNTrees(nt: Int): this.type = {
    require(nt > 1)
    this.nTrees = nt
    this
  }
  
  def setOverlapDistance(od: Double): this.type = {
    this.overlapDistance = od
    this
  }
  
  def setSampleSizes(ss: Array[Int]): this.type = {
    require(ss.length > 1 && ss.forall(_ > 0))
    this.sampleSizes = ss
    this
  }  
  
  /**
   * Set the seed for the sampling mechanism.
   */
  def setSeed(s: Long): this.type = {
    this.seed = s
    this
  }
  
  def setInitialRDD(rdd: RDD[LabeledPoint]): this.type = {
    this.initialRDD = rdd
    this
  }

  /**
   * Update the clustering model by training on batches of data from a DStream.
   * This operation registers a DStream for training the model,
   * checks whether the cluster centers have been initialized,
   * and updates the model using each batch of data from the stream.
   *
   * @param data DStream containing vector data
   */
  def trainOn(data: DStream[LabeledPoint]) {
    val sc = data.context.sparkContext
    val queue = new Queue[LabeledPoint]
    var nelem = 0L
    
    data.foreachRDD { (rdd, time) =>
      
      if(model.topTree == null) {   
        // There is no master tree created yet
        val csize = rdd.count()
        nelem += csize
        if(nelem >= nTrees) {
          // Enough elements to create the master and the sub-trees
          val sc = rdd.context
          val firstRDD = if(queue.size > 0) rdd.union(sc.parallelize(queue, nTrees)) else rdd
          model = initializeModel(firstRDD)
          logInfo("Number of instances in the new case-base: " + model.trees.map(_.getSize).sum.toInt) // Important
          queue.clear()
          nelem = 0
        } else {          
          queue ++= rdd.collect()
        }
      } else {
        if(isUnbalanced()){ // No re-balance for static experiments 
          logInfo("Re-balancing the distributed m-tree." + 
              "One or more sub-trees have grown too much.")
          // Re-balance the whole case-base. New topTree and a new re-partition process is started for sub-trees
          val casebase = model.trees.flatMap{ tree =>
            tree.getIterator.map(jlp => LPUtils.fromJavaLP(jlp)) 
          }
          // Swap model
          val oldModel = model
          model = initializeModel(casebase)
          logInfo("Number of instances in the new case-base: " + model.trees.map(_.getSize).sum.toInt) // Important
          //oldModel.trees.unpersist()
          nelem = 0
        }
        // Insert new examples
        if (!rdd.isEmpty()){
          // Swap model
          val oldModel = model
          model = if(edited) editModel(rdd) else insertNewExamples(rdd)
          logInfo("Number of instances in the modified case-base: " + model.trees.map(_.getSize).sum.toInt) // Important
          //oldModel.trees.unpersist()
        }
      }
    }
  }
  
  private def isUnbalanced() = {
    if(model != null){
      !model.trees.filter(_.getSize < kGraph * 2).isEmpty()
    } else {
      false
    }
  }
  
  private def initializeModel(rdd: RDD[LabeledPoint]) = {
      // Initialize topTree (first registers coming...)
      val rand = new XORShiftRandom(this.seed + 2)
      val firstLoad = rdd.takeSample(false, nTrees, rand.nextLong())
      val bTopTree = rdd.context.broadcast(new MTreeWrapper(firstLoad.zipWithIndex.map(lp => LPUtils.toIndexedJavaLP(lp._1, lp._2))))
      
      val tau = if(overlapDistance < 0) {
        estimateTau(rdd.map(lp => 
          new VectorWithNorm(lp.features)), sampleSizes, seed)
      } else {
        overlapDistance
      }        
      logInfo("Tau is: " + tau)
      
      // Load the instances in the sub-trees
      val repartitioned = rdd.map(v => (v, null)).partitionBy(new KNNPartitioner(bTopTree))
      val trees = repartitioned.mapPartitions { itr =>
        val childTree = new MTreeWrapper(itr.map(t => LPUtils.toJavaLP(t._1)).toArray)
        Iterator(childTree)
      }.persist(StorageLevel.MEMORY_AND_DISK)
        
      new StreamingDistributedKNNModel(bTopTree, trees, tau)
  }
  
  private def editModel(rdd: RDD[LabeledPoint]): StreamingDistributedKNNModel = {
    
      val newTrees = if(model.tau > 0) {
        val localGraph = model.accuratekNNQuery(rdd, kGraph)
        val edited = InstanceSelection.instanceSelection(localGraph, removeOld)        
        editTrees(edited, model.trees).persist(StorageLevel.MEMORY_AND_DISK)
      } else {
        model.fastEdition(rdd, kGraph, removeOld).persist(StorageLevel.MEMORY_AND_DISK)
      }      
      new StreamingDistributedKNNModel(model.topTree, newTrees, model.tau)
  }
  
  private def editTrees(rdd: RDD[TreeLP], trees: RDD[MTreeWrapper]) = {
      val searchData = rdd.map(tp => tp.itree -> tp).partitionBy(new HashPartitioner(trees.partitions.length))
      searchData.zipPartitions(trees) {
        (childData, trees) =>
          val tree = trees.next()
          assert(!trees.hasNext)
          childData.foreach { case (_, lp) =>
            if(lp.action == INSERT)
              tree.insert(LPUtils.toJavaLP(lp.point))
            else if (lp.action == REMOVE)
              tree.remove(LPUtils.toJavaLP(lp.point))
          }
          Iterator(tree)
      }      
  }
  
  private def removeExamples(rdd: RDD[(Int, LabeledPoint)], trees: RDD[MTreeWrapper]) = {
      val searchData = rdd.partitionBy(new HashPartitioner(trees.partitions.length))
      searchData.zipPartitions(trees) {
        (childData, trees) =>
          val tree = trees.next()
          assert(!trees.hasNext)
          childData.foreach { case (index, lp) =>
            tree.remove(LPUtils.toJavaLP(lp))
          }
          Iterator(tree)
      }      
  }
  
  private def insertNewExamples(rdd: RDD[LabeledPoint]) = {      
      // Add new examples to the casebase
      val searchData = rdd.map { vector =>
          // Just use one index for insertion (exact insertion is not relevant)
          val index = StreamingDistributedKNN.searchIndex(vector.features, model.topTree.value)
          assert(index != -1, s"indices must be non-empty: $vector")
          index -> vector
      }.partitionBy(new HashPartitioner(model.trees.partitions.length))
  
      // for each partition, search points within corresponding child tree
      val newTrees = searchData.zipPartitions(model.trees) {
        (childData, trees) =>
          val tree = trees.next()
          assert(!trees.hasNext)
          childData.foreach { case (index, lp) =>
            tree.insert(LPUtils.toJavaLP(lp))
          }
          Iterator(tree)
      }.persist(StorageLevel.MEMORY_AND_DISK)
        
      new StreamingDistributedKNNModel(model.topTree, newTrees, model.tau)
  }

  /**
   * Java-friendly version of `trainOn`.
   */
  def trainOn(data: JavaDStream[LabeledPoint]): Unit = trainOn(data.dstream)

  /**
   * Use the model to make predictions on the values of a DStream and carry over its keys.
   *
   * @param data DStream containing (key, feature vector) pairs
   * @tparam K key type
   * @return DStream containing the input keys and the predictions as values (label, prediction)
   */
  def predictOnValues(data: DStream[LabeledPoint], k: Int): DStream[(Double, Double)] = {
    //assertInitialized()
    data.transform(rdd => model.predict(rdd, k))
  }

  /**
   * Java-friendly version of `predictOnValues`.
   */
  def predictOnValues(
      data: JavaDStream[LabeledPoint], k: Int): JavaPairDStream[Double, Double] = {
    JavaPairDStream.fromPairDStream(
      predictOnValues(data.dstream, k).asInstanceOf[DStream[(Double, Double)]])
  }
  
  /** Check whether cluster centers have been initialized. */
  private[this] def assertInitialized(): Boolean = {
    model.topTree == null
  }
}

object LPUtils {
  def toJavaLP(lp: LabeledPoint) = {
    new DataLP(lp.features.toArray.map(_.toFloat), lp.label.toFloat)
  }
  
  def toIndexedJavaLP(lp: LabeledPoint, index: Int) = {
    new IndexedLP(lp.features.toArray.map(_.toFloat), lp.label.toFloat, index)
  }
  
  def fromJavaLP(jlp: DataLP) = {
    new LabeledPoint(jlp.getLabel.toDouble, Vectors.dense(jlp.getFeatures.map(_.toDouble)))
  }  
}

object StreamingDistributedKNN {
  
  trait Action
  case object INSERT extends Action
  case object REMOVE extends Action
  case object NOINSERT extends Action
  case object NOREMOVE extends Action
  case class TreeLP(point: LabeledPoint, itree: Int, action: Action)
  
  /**
    * Estimate a suitable buffer size based on dataset
    *
    * A suitable buffer size is the minimum size such that nearest neighbors can be accurately found even at
    * boundary of splitting plane between pivot points. Therefore assuming points are uniformly distributed in
    * high dimensional space, it should be approximately the average distance between points.
    *
    * Specifically the number of points within a certain radius of a given point is proportionally to the density of
    * points raised to the effective number of dimensions, of which manifold data points exist on:
    * R_s = \frac{c}{N_s ** 1/d}
    * where R_s is the radius, N_s is the number of points, d is effective number of dimension, and c is a constant.
    *
    * To estimate R_s_all for entire dataset, we can take samples of the dataset of different size N_s to compute R_s.
    * We can estimate c and d using linear regression. Lastly we can calculate R_s_all using total number of observation
    * in dataset.
    *
    */
  def estimateTau(data: RDD[VectorWithNorm], sampleSize: Array[Int], seed: Long): Double = {
    val total = data.count()

    // take samples of points for estimation
    val samples = data.mapPartitionsWithIndex {
      case (partitionId, itr) =>
        val rand = new XORShiftRandom(byteswap64(seed ^ partitionId))
        itr.flatMap {
          p => sampleSize.zipWithIndex
            .filter { case (size, _) => rand.nextDouble() * total < size }
            .map { case (size, index) => (index, p) }
        }
    }
    // compute N_s and R_s pairs
    val estimators = samples
      .groupByKey()
      .map {
        case (index, points) => (points.size, computeAverageDistance(points))
      }.collect().distinct

    // collect x and y vectors
    val x = BDV(estimators.map { case (n, _) => math.log(n) })
    val y = BDV(estimators.map { case (_, d) => math.log(d) })

    // estimate log(R_s) = alpha + beta * log(N_s)
    val xMeanVariance: MeanAndVariance = meanAndVariance(x)
    val xmean = xMeanVariance.mean
    val yMeanVariance: MeanAndVariance = meanAndVariance(y)
    val ymean = yMeanVariance.mean

    val corr = (mean(x :* y) - xmean * ymean) / math.sqrt((mean(x :* x) - xmean * xmean) * (mean(y :* y) - ymean * ymean))

    val beta = corr * yMeanVariance.stdDev / xMeanVariance.stdDev
    val alpha = ymean - beta * xmean
    val rs = math.exp(alpha + beta * math.log(total))

    // c = alpha, d = - 1 / beta
    rs / math.sqrt(-1 / beta)
  }

  // compute the average distance of nearest neighbors within points using brute-force
  private[this] def computeAverageDistance(points: Iterable[VectorWithNorm]): Double = {
    val distances = points.map {
      point => points.map(p => p.fastSquaredDistance(point)).filter(_ > 0).min
    }.map(math.sqrt)

    distances.sum / distances.size
  }

  
  def searchIndex(v: Vector, tree: MTreeWrapper): Int = {
    val neigs = tree.kNNQuery(1, LPUtils.toJavaLP(new LabeledPoint(0, v)).getFeatures)
    if(neigs.isEmpty()) -1 else neigs.get(0).data.asInstanceOf[IndexedLP].getIndex
  }
  
  def searchIndices(v: Vector, tree: MTreeWrapper, tau: Double): Seq[Int] = {
    val neigs =  tree.searchIndices(LPUtils.toJavaLP(new LabeledPoint(0, v)).getFeatures, tau)
    if(neigs.isEmpty()) Seq(-1) else neigs.map(_.data.asInstanceOf[IndexedLP].getIndex).toSeq
  }
}

/**
* Partitioner used to map vector to leaf node which determines the partition it goes to
*
* @param tree [[MetricTree]] used to find leaf
*/
class KNNPartitioner(treeIndex: Broadcast[MTreeWrapper]) extends Partitioner {
  override def numPartitions: Int = treeIndex.value.getIterator.map(_.asInstanceOf[IndexedLP].getIndex).max + 1

  override def getPartition(key: Any): Int = {
    key match {
      case v: LabeledPoint => StreamingDistributedKNN.searchIndex(v.features, treeIndex.value)
      case _ => throw new IllegalArgumentException(s"Key must be of type LabeledPoint but got: $key")
    }
  }
}

