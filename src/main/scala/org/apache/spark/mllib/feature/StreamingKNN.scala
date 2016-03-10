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

package org.apache.spark.mllib.clustering

import scala.reflect.ClassTag

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
import xxl.core.indexStructures.MTree._
import xxl.core.collections.containers.CounterContainer
import xxl.core.collections.containers.io._
import xxl.core.indexStructures.mtrees.MTreeTest
import collection.JavaConversions._
import org.apache.spark.streaming.Time

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
@Since("1.2.0")
class StreamingKNNModel (val casebase: RDD[(Int, MTreeTest)], val size: Double) extends Serializable with Logging {

  /**
   * Perform a k-means update on a batch of data.
   */  
  def predict[K: ClassTag](data: RDD[LabeledPoint], k: Int) = {
    
    val bdata = data.context.broadcast(data.zipWithIndex().map(_.swap).collect())
    println("casebase size: " + casebase.map{ case (_, t) => t.getSize }.sum)
    val preds = casebase.flatMap{ case(_, tree) => 
      val labelsAndPres = bdata.value.map{ case(idx, lp) => 
        val dls = tree.kNNQuery(k, lp.features.toArray.map(_.toFloat))
            .map(t => (t.getDistance.toFloat, t.getLabel.toFloat))
            .sortBy(-_._1)
            .toArray
        idx -> (lp.label.toFloat, dls)
      }
      labelsAndPres
    }
    
    val count = preds.count()
    if(count > 0) { println("preds: " + preds.first()._2._2.mkString(",")); println("preds count: " + count) }
    
    /* Reduce the neighbors to get the top K from all the trees (buffers must be sorted!!) */
    preds.reduceByKey{ case ((l1, b1), (_, b2)) => 
      val res = Array.ofDim[(Float, Float)](k)
      var i = 0; var j = 0; var c = 0
      // Get the biggest elements from both buffers
      while(c < k && i < b1.size && j < b2.size) {
        if(b1(i)._1 > b2(j)._1) {
          res(c) = b1(i)
          c += 1; i += 1
        } else {
          res(c) = b2(j)
          c += 1; j += 1
        }
      }
      // Add the remaining elements from a single buffer
      while(i < b1.size && c < k) { res(c) = b1(i); c += 1; i += 1  }
      while(j < b2.size && c < k) { res(c) = b2(j); c += 1; j += 1  }
      l1 -> res.filter(_ != null)
    }.map { case (_, (label, a)) =>
      println("a printed: " + a.mkString(","))
      val pred = a.map(_._2).groupBy(identity).maxBy(_._2.size)._1
      (label, pred) 
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
@Since("1.2.0")
class StreamingKNN @Since("1.2.0") (
    @Since("1.2.0") var k: Int,
    @Since("1.2.0") var nPartitions: Int) extends Logging with Serializable {

  @Since("1.2.0")
  def this() = this(2, 5)

  protected var model: StreamingKNNModel = new StreamingKNNModel(null, 0.0)

  /**
   * Set the number of clusters.
   */
  @Since("1.2.0")
  def setK(k: Int): this.type = {
    this.k = k
    this
  }

  /**
   * Set the forgetfulness of the previous centroids.
   */
  @Since("1.2.0")
  def setNPartitions(np: Int): this.type = {
    this.nPartitions = np
    this
  }

  /**
   * Return the latest model.
   */
  @Since("1.2.0")
  def latestModel(): StreamingKNNModel = {
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
  def trainOn(data: DStream[LabeledPoint], dimension: Int) {
    val sc = data.context.sparkContext
    val trees = (0 until nPartitions).map(i => i -> new MTreeTest(i, dimension))
    model = new StreamingKNNModel(sc.parallelize(trees, nPartitions), 0)
    data.foreachRDD { (rdd, time) =>
      val irdd = rdd.map(v => scala.util.Random.nextInt(nPartitions) -> v).groupByKey()
      val count = irdd.map(_._2.size).sum
      println("Indexed elements: " + count)
      model = if(count > 0) {
        val updated = irdd.join(model.casebase).map{ case (idx, (ps, tree)) =>  
          for(point <- ps) {
            tree.insert(point.features.toArray.map(_.toFloat), point.label.toFloat)
          }
          //ps.foreach (p => tree.insert(p.features.toArray.map(_.toFloat), p.label.toFloat))
          idx -> tree    
        }
        val size = updated.map{ case (_, t) => t.getSize }.sum
        println("Model size: " + size)
        new StreamingKNNModel(updated, size)
      } else {
        model 
      }
      //println("Number of cases in the new model: " + updated.map(_.getSize).sum)
      //model = new StreamingKNNModel(updated)
    }
    //model.casebase.cache()
    //println("Number of cases in the new model: " + updateModel.map(_._2.getSize).sum)
  }

  /**
   * Java-friendly version of `trainOn`.
   */
  @Since("1.4.0")
  def trainOn(data: JavaDStream[LabeledPoint]): Unit = trainOn(data.dstream)

  /**
   * Use the clustering model to make predictions on batches of data from a DStream.
   *
   * @param data DStream containing vector data
   * @return DStream containing predictions
   */
  /*def predictOn(data: DStream[Vector]): DStream[Float] = {
    assertInitialized()
    data.map(rdd => model.predict(rdd, k))
  }*/

  /**
   * Java-friendly version of `predictOn`.
   */
  /*def predictOn(data: JavaDStream[Vector]): JavaDStream[java.lang.Integer] = {
    JavaDStream.fromDStream(predictOn(data.dstream).asInstanceOf[DStream[java.lang.Integer]])
  }*/

  /**
   * Use the model to make predictions on the values of a DStream and carry over its keys.
   *
   * @param data DStream containing (key, feature vector) pairs
   * @tparam K key type
   * @return DStream containing the input keys and the predictions as values (label, prediction)
   */
  def predictOnValues(data: DStream[LabeledPoint]): DStream[(Float, Float)] = {
    //assertInitialized()
    data.transform(rdd => model.predict(rdd, k))
  }

  /**
   * Java-friendly version of `predictOnValues`.
   */
  /*Stream.fromPairDStream(
      predictOnValues(data.dstream).asInstanceOf[DStream[(K, java.lang.Integer)]])
  }*/
  
  /** Check whether cluster centers have been initialized. */
  private[this] def assertInitialized(): Unit = {
    if (model.size == 0) {
      throw new IllegalStateException(
        "Initial case-base must be set before starting predictions")
    }
  }
}