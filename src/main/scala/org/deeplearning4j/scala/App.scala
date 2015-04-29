package org.deeplearning4j.scala

import org.canova.image.recordreader.ImageRecordReader
import org.canova.api.split.FileSplit
import java.io.File
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import java.util
import org.nd4j.linalg.factory.Nd4j
import au.com.bytecode.opencsv.{CSVReadProc, CSV}

/**
 * Created by sogolmoshtaghi on 4/21/15.
 */
object App {

  def main(args: Array[String]) {
    val recordReader = new ImageRecordReader(56, 56, true)
    val home = System.getProperty("user.home")
    recordReader.initialize(new FileSplit(new File(home, "lfw")))


    val iterator = new RecordReaderDataSetIterator(recordReader, 100)

    while (iterator.hasNext) {
      val dataset = iterator.next()
      dataset.normalizeZeroMeanZeroUnitVariance()
      val features = dataset.getFeatureMatrix()
      println(util.Arrays.toString(features.shape()))
      val first = features.slice(0)
      println(first)

    }

  }
}
