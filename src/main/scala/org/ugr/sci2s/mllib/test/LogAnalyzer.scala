package org.ugr.sci2s.mllib.test

import scala.io.Source
import java.io.PrintWriter
import java.io.File
import java.io.BufferedWriter
import java.io.FileWriter

object LogAnalyzer {

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
    
    val dirName = params.getOrElse("dirName", "/home/sramirez/git/spark-IS-streaming/logs/")
    val files = new java.io.File(dirName).listFiles.filter(_.getName.endsWith(".out"))
    
    for(jfile <- files){
      val fileName: String = jfile.getName
      val outputFile = fileName.split("//").last + ".csv"
      val regReduce = "reduce at QueuRDDStreamingTest.scala:(\\d{2,})\\) finished in".r
      val linesToSelect = (line: String) => {
        regReduce.findFirstIn(line).isDefined || // reduce phase (classify)
        line.contains("Accuracy per batch") || 
        line.contains("map at StreamingDistributedKNN.scala:165) finished") || // fast knn query (classify) 
        line.contains("map at StreamingDistributedKNN.scala:427) finished") || // insert new examples (training)       
        line.contains("map at StreamingDistributedKNN.scala:189) finished") || // fast edition (training)
        line.contains("sum at StreamingDistributedKNN.scala:375) finished") || // final action (training)
        line.contains("Number of instances in the modified case-base") ||
        line.contains("Accuracy per batch") ||        
        line.contains("Batch scheduling delay") ||
        line.contains("Batch processing time")
      }
      
      val lines = Source.fromFile(jfile.getAbsolutePath).getLines.toArray.zipWithIndex.filter{case (line, _) => linesToSelect(line)}
      val nbatches = lines.filter(_._1.contains("Batch processing time")).length
      val trainTime = Array.fill[Float](nbatches)(0)
      val classificTime = Array.fill[Float](nbatches)(0)
      val accuracy = Array.fill[Float](nbatches)(0)
      val instances = Array.fill[Int](nbatches)(0)
      val delays = Array.fill[Float](nbatches)(0)
      val batchTime = Array.fill[Float](nbatches)(0)
      var trainingIndex = 0; var clsIndex = 0; var bIndex = 0;
      
      println("Num. batches: " + nbatches)
      for((line, _) <- lines) {
        val tokens = line.split(" ")
        if(line.contains("map at StreamingDistributedKNN.scala:189) finished")) {
          classificTime(clsIndex) += tokens(tokens.length - 2).replace(',', '.').toFloat
        } else if(regReduce.findFirstIn(line).isDefined) {
          classificTime(clsIndex) += tokens(tokens.length - 2).replace(',', '.').toFloat
        } else if(line.contains("map at StreamingDistributedKNN.scala:165) finished") || line.contains("map at StreamingDistributedKNN.scala:427) finished")) {
          trainTime(trainingIndex) += tokens(tokens.length - 2).replace(',', '.').toFloat
        } else if(line.contains("sum at StreamingDistributedKNN.scala:375) finished")) {
          trainTime(trainingIndex) += tokens(tokens.length - 2).replace(',', '.').toFloat
        } else if(line.contains("Number of instances in the modified case-base")) {
          instances(trainingIndex) = tokens(tokens.length - 1).toFloat.toInt           
          trainingIndex += 1
        } else if (line.contains("Accuracy per batch")) {
          accuracy(clsIndex) = tokens(tokens.length - 1).toFloat  
          clsIndex +=1
        } else if (line.contains("Batch scheduling delay")) {
          delays(bIndex) = tokens(tokens.length - 1).toFloat        
        } else if (line.contains("Batch processing time")) {
          batchTime(bIndex) = tokens(tokens.length - 1).toFloat
          bIndex += 1
        }        
      }
      
      val newsize = trainTime.filter(_ != 0).length
      val avg = accuracy.filter(_ != 0).sum / accuracy.filter(_ != 0).length
      val avgtrtime = trainTime.filter(_ != 0).sum / trainTime.filter(_ != 0).length
      val avgclstime = classificTime.filter(_ != 0).sum / classificTime.filter(_ != 0).length
      val avginstances = instances.slice(0, newsize).filter(_ != 0).sum / instances.slice(0, newsize).filter(_ != 0).length
      
      val output = "Training time," + trainTime.slice(0, newsize).mkString(",") + "\n" + 
        "Classification time," + classificTime.slice(0, newsize).mkString(",") + "\n" +
        "Scheduling delay," + delays.slice(0, newsize).mkString(",") + "\n" +
        "Batch time," + batchTime.slice(0, newsize).mkString(",") + "\n" +        
        "Accuracy," + accuracy.slice(0, newsize).mkString(",") + "\n" + 
        "Instances, " + instances.slice(0, newsize).mkString(",") + "\n" +
        "Avg. acc," + avg + "\n" +        
        "Avg. train time," + avgtrtime + "\n" +        
        "Avg. cls time," + avgclstime + "\n" +  
        "Avg. instances," + avginstances + "\n" +  
        "Total batches," + newsize + "\n"
        
      println(output)
      val file = new File(outputFile)
      val bw = new BufferedWriter(new FileWriter(file))
      bw.write(output)
      bw.close()
    }
    
    
  }

}