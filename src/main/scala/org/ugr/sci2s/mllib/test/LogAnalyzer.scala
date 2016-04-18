package org.ugr.sci2s.mllib.test


import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.StreamingDistributedKNN
import org.apache.spark.mllib.feature.StreamingDistributedKNN._
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.scheduler.StreamingListener
import org.apache.spark.streaming.scheduler.StreamingListenerBatchCompleted
import org.apache.spark.streaming.scheduler.StreamingListenerBatchStarted
import org.apache.spark.streaming.scheduler.StreamingListenerBatchCompleted
import mtree.DataLP
import mtree.IndexedLP
import org.apache.spark.mllib.knn.VectorWithNorm
import mtree.MTree
import org.apache.spark.mllib.linalg.Vectors
import scala.io.Source
import java.io.PrintWriter
import java.io.File
import java.io.BufferedWriter
import java.io.FileWriter

object LogAnalyzer extends Logging {

  def main(args: Array[String]): Unit = {   
    
    // Create a table of parameters (parsing)
    val params = args.map({arg =>
        val param = arg.split("--|=").filter(_.size > 0)
        param.size match {
          case 2 =>  (param(0) -> param(1))
          case _ =>  ("" -> "")
        }
    }).toMap  
    
    println("Parameters: " + params.toString())
    
    val dirName = params.getOrElse("dirName", "/home/sramirez/git/spark-IS-streaming/")
    val files = new java.io.File(dirName).listFiles.filter(_.getName.endsWith(".out"))
    
    for(file <- files){
      val fileName = file.getName
      val outputFile = fileName.split("//").last + ".csv"
      val regReduce = "reduce at QueuRDDStreamingTest.scala:(\\d{2,})\\) finished in".r
      val linesToSelect = (line: String) => {
        regReduce.findFirstIn(line).isDefined || 
        line.contains("Accuracy per batch") ||
        line.contains("map at StreamingDistributedKNN.scala:137") ||
        line.contains("map at StreamingDistributedKNN.scala:161") ||
        line.contains("sum at StreamingDistributedKNN.scala:344") ||  
        line.contains("Number of instances in the modified case-base")
      }
      
      val lines = file.getLines.toArray.zipWithIndex.filter{case (line, _) => linesToSelect(line)}
      val nbatches = lines.filter(_._1.contains("Number of instances in the modified case-base")).length
      val trainTime = Array.fill[Float](nbatches)(0)
      val classificTime = Array.fill[Float](nbatches)(0)
      val accuracy = Array.fill[Float](nbatches)(0)
      val instances = Array.fill[Int](nbatches)(0)
      var trainIndex = 0; var classificIndex = 0
      var index = 0;
      
      println("Num. batches: " + nbatches)
      for((line, _) <- lines) {
        val tokens = line.split(" ")
        if(line.contains("map at StreamingDistributedKNN.scala:137")) {
          trainTime(index) += tokens(tokens.length - 2).replace(',', '.').toFloat
        } else if(regReduce.findFirstIn(line).isDefined) {
          trainTime(index) += tokens(tokens.length - 2).replace(',', '.').toFloat
        } else if(line.contains("map at StreamingDistributedKNN.scala:161")) {
          classificTime(index) += tokens(tokens.length - 2).replace(',', '.').toFloat
        } else if(line.contains("sum at StreamingDistributedKNN.scala:344")) {
          classificTime(index) += tokens(tokens.length - 2).replace(',', '.').toFloat
        } else if(line.contains("Number of instances in the modified case-base")) {
          instances(index) = tokens(tokens.length - 1).toFloat.toInt
          index += 1
        } else if (line.contains("Accuracy per batch")) {
          accuracy(index) = tokens(tokens.length - 1).toFloat        
        }
        
      }
      
      val output = "Training time," + trainTime.mkString(",") + "\n" + 
        "Classification time," + classificTime.mkString(",") + "\n" +
        "Accuracy," + accuracy.mkString(",") + "\n" + 
        "Total acc," + accuracy.sum / accuracy.length + "\n" +
        "Instances, " + instances.mkString(",") + "\n"
  
      println(output)
      val file = new File(outputFile)
      val bw = new BufferedWriter(new FileWriter(file))
      bw.write(output)
      bw.close()
    }
    
    
  }

}