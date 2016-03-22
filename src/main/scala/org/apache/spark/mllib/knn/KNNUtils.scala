package org.apache.spark.mllib.knn

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLUtils
import breeze.linalg.{DenseVector => BDV, Vector => BV}
import org.apache.spark.mllib.linalg.Vectors

object KNNUtils {
  def fastSquaredDistance(
                           v1: Vector,
                           norm1: Double,
                           v2: Vector,
                           norm2: Double,
                           precision: Double = 1e-6): Double =
    MLUtils.fastSquaredDistance(v1, norm1, v2, norm2, precision)
}

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