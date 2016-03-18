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
import xxl.core.indexStructures.MTree._
import xxl.core.collections.containers.CounterContainer
import xxl.core.collections.containers.io._
import xxl.core.indexStructures.mtrees.MTreeLP
import collection.JavaConversions._
import scala.collection.mutable.Queue
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
import org.apache.spark.streaming.Time
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.Partitioner
import org.apache.spark.rdd.ShuffledRDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import breeze.stats._
import org.apache.spark.mllib.knn.KNNUtils
import breeze.linalg.{DenseVector => BDV, Vector => BV}
import org.apache.spark.mllib.feature.StreamingDistributedKNN.VectorWithNorm


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
    val topTree: Broadcast[MTreeLP],
    val trees: RDD[MTreeLP],
    val indexMap: Map[Vector, Int],
    val tau: Double) extends Serializable with Logging {
  
  def kNNQuery(data: RDD[LabeledPoint], k: Int): RDD[(LabeledPoint, Array[(LabeledPoint, Float)])] = {
    if(topTree != null) {
      val searchData = data.flatMap { vector =>
          val indices = StreamingDistributedKNN.searchIndices(vector.features, topTree.value, indexMap, tau)
            .map(i => (i, vector))

          assert(indices.filter(_._1 == -1).length == 0, s"indices must be non-empty: $vector")
          indices
      }.partitionBy(new HashPartitioner(trees.partitions.length))

      val asd = searchData.count()
      println("Indices count: " + asd)
      // for each partition, search points within corresponding child tree
      val results = searchData.zipPartitions(trees) {
        (childData, trees) =>
          val tree = trees.next()
          assert(!trees.hasNext)
          childData.flatMap {
            case (_, point) =>
              tree.kNNQuery(k, point.features.toArray.map(_.toFloat))
                  .map(pair => (point, (pair.getPoint, pair.getDistance)))
              /*{
                case (neighbor, distance) if distance <= $(maxDistance) =>
                  (i, (neighbor.row, distance))
              }*/
          }
      }
  
      // merge results by point index together and keep topK results
      results.topByKey(k)(Ordering.by(-_._2))
        .map { case (point, seq) => (point, seq.map(e => LPUtils.fromJavaLP(e._1) -> e._2.floatValue())) }
    } else {
      data.context.emptyRDD[(LabeledPoint, Array[(LabeledPoint, Float)])]
    }
  }
  
  def predict(data: RDD[LabeledPoint], k: Int): RDD[(Double, Double)] = {
    if(topTree != null) {
      kNNQuery(data, k).map{ case (lp, arr) =>
        val pred = arr.map(_._1.label).groupBy(identity).maxBy(_._2.size)._1
        (lp.label, pred)
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
    var nPartitions: Int,
    var sampleSizes: Array[Int],
    var seed: Long) extends Logging with Serializable {

  def this() = this(2, (100 to 1000 by 100).toArray, 26827651492L)

  protected var model: StreamingDistributedKNNModel = new StreamingDistributedKNNModel(null, null, null, 0.0)

  /**
   * Set the number of partitions/sub-trees to use in the model.
   */
  def setNPartitions(np: Int): this.type = {
    this.nPartitions = np
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

  /**
   * Return the latest model.
   */
  def updatedModel(): StreamingDistributedKNNModel = {
    model
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
    
    data.foreachRDD { (rdd, time) =>
      
      if(model.topTree == null) {   
        // There is no master tree created yet
        if(rdd.count() >= nPartitions) {
          // Enough elements to create the master and the sub-trees
          val sc = rdd.context
          val firstRDD = if(queue.size > 0) rdd.union(sc.parallelize(queue, nPartitions)) else rdd
          model = initializeModel(firstRDD)
          queue.clear()
        } else {          
          queue ++= rdd.collect()
        }
      } else if (isUnbalanced) {
        // Re-balance the whole case-base. New topTree and a new re-partition process is started for sub-trees
        val casebase = model.trees.flatMap{ tree =>
          tree.getIterator.map(jlp => LPUtils.fromJavaLP(jlp)) 
        }
        model = initializeModel(casebase) // new model re-build using the old case-base
        model = insertNewExamples(rdd) // add the new examples to the re-balanced case-base
      } else {
        model = insertNewExamples(rdd)
      }
      //val s = model.trees.map(_.getSize).sum()
      //println("Model size: " + s)      
    }
  }
  
  private def isUnbalanced() = {
    if(model != null){
      val sizes = model.trees.map(_.getSize)
      sizes.max > sizes.min * 2 
    } else {
      false
    }
  }
  
  private def initializeModel(rdd: RDD[LabeledPoint]) = {
      // Initialize topTree (first registers coming...)
      val rand = new XORShiftRandom(this.seed)
      val firstLoad = rdd.takeSample(false, this.nPartitions, rand.nextLong())
      val indexMap = firstLoad.map(_.features).zipWithIndex.toMap
      val bulkFirst = firstLoad.map( lp => (LPUtils.toJavaLP(lp), lp.features(0))).sortBy(_._2).map(_._1)
      //topTree.bulkInsert(bulkFirst)
      val topTree = new MTreeLP(bulkFirst)
      //val topTree = new MTreeLP()
      //rdd.collect.foreach (lp => topTree.insert(LPUtils.toJavaLP(lp)))
      val bTopTree = rdd.context.broadcast(topTree)
      
      val tau = StreamingDistributedKNN.estimateTau(rdd.map(lp => new VectorWithNorm(lp.features)), sampleSizes, seed)
      logInfo("Tau is: " + tau)
      
      // Load the instances in the sub-trees
      val repartitioned = new ShuffledRDD(rdd.map(v => (v, null)), new KNNPartitioner(bTopTree, indexMap))
      val trees = repartitioned.mapPartitions { itr =>
        // Sort elements by first dimension before bulk-loading
        val bulkElements = itr.toArray.map{ case(lp, _) => (lp, lp.features(0))}.sortBy(_._2).map(_._1)
        val childTree = new MTreeLP(bulkElements.map(LPUtils.toJavaLP))
        Iterator(childTree)
      }.persist(StorageLevel.MEMORY_AND_DISK)
        
      new StreamingDistributedKNNModel(bTopTree, trees, indexMap, tau)
  }
  
  private def insertNewExamples(rdd: RDD[LabeledPoint]) = {
      // Add new examples to the casebase
      val searchData = rdd.map { vector =>
          // Just use one index for insertion (exact insertion is not relevant)
          val index = StreamingDistributedKNN.searchIndex(vector.features, model.topTree.value, model.indexMap)
          assert(index == -1, s"indices must be non-empty: $vector")
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
      
      // New estimation of TAU
      val allExamples = model.trees.flatMap(_.getIterator)
      val tau = StreamingDistributedKNN.estimateTau(allExamples.map(lp => 
        new VectorWithNorm(LPUtils.fromJavaLP(lp).features)), sampleSizes, seed)
      
      new StreamingDistributedKNNModel(model.topTree, newTrees, model.indexMap, tau)
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
    new xxl.core.spatial.points.LabeledPoint(lp.features.toArray.map(_.toFloat), lp.label.toFloat)
  }
  
  def fromJavaLP(jlp: xxl.core.spatial.points.LabeledPoint) = {
    new LabeledPoint(jlp.getLabel.toDouble, Vectors.dense(jlp.getFeatures.map(_.toDouble)))
  }  
}

object StreamingDistributedKNN {
  
    /**
    * VectorWithNorm can use more efficient algorithm to calculate distance
    */
  case class VectorWithNorm(vector: Vector, norm: Double) {
    def this(vector: Vector) = this(vector, Vectors.norm(vector, 2))

    def this(vector: BV[Double]) = this(Vectors.fromBreeze(vector))

    def fastSquaredDistance(v: VectorWithNorm): Double = {
      KNNUtils.fastSquaredDistance(vector, norm, v.vector, v.norm)
    }

    def fastDistance(v: VectorWithNorm): Double = math.sqrt(fastSquaredDistance(v))
  }
  
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

  
  def searchIndex(v: Vector, tree: MTreeLP, index: Map[Vector, Int]): Int = {
    val jlp = tree.kNNQuery(1, LPUtils.toJavaLP(new LabeledPoint(0, v)).getFeatures)(0).getPoint        
    index.getOrElse(LPUtils.fromJavaLP(jlp).features, -1)
  }
  
  // We get all the sub-trees that overlap with our point
  def searchIndices(v: Vector, tree: MTreeLP, index: Map[Vector, Int]): Seq[Int] = {
    val jlps = tree.overlapQuery(LPUtils.toJavaLP(new LabeledPoint(0, v)).getFeatures)
    jlps.map(jlp => index.getOrElse(LPUtils.fromJavaLP(jlp).features, -1)).toSeq
  }
  
  // We get all the sub-trees that overlap with our point
  def searchIndices(v: Vector, tree: MTreeLP, index: Map[Vector, Int], tau: Double): Seq[Int] = {
    val jlps = tree.overlapQuery(LPUtils.toJavaLP(new LabeledPoint(0, v)).getFeatures, tau)
    jlps.map(jlp => index.getOrElse(LPUtils.fromJavaLP(jlp).features, -1)).toSeq
  }
}

/**
* Partitioner used to map vector to leaf node which determines the partition it goes to
*
* @param tree [[MetricTree]] used to find leaf
*/
class KNNPartitioner(treeIndex: Broadcast[MTreeLP], indexMap: Map[Vector, Int]) extends Partitioner {
  override def numPartitions: Int = indexMap.size

  override def getPartition(key: Any): Int = {
    key match {
      case v: LabeledPoint => StreamingDistributedKNN.searchIndex(v.features, treeIndex.value, indexMap)
      case _ => throw new IllegalArgumentException(s"Key must be of type LabeledPoint but got: $key")
    }
  }
}