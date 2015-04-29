package org.deeplearning4j.scala

import org.canova.api.split.FileSplit
import java.io.File
import org.canova.sound.recordreader.WavFileRecordReader
import org.canova.common.RecordConverter

/**
 * Created by sogolmoshtaghi on 4/21/15.
 */
object AudioVectorizer {

  def main(args: Array[String]) {
    val recordReader = new WavFileRecordReader(true)
    val filePath = System.getProperty("user.home") + "/dsr/data/mlsp_contest_dataset/essential_data/src_wavs"

    recordReader.initialize(new FileSplit(new File(filePath)))

    while (recordReader.hasNext()) {
      val dataset = recordReader.next()
      val arr = RecordConverter.toArray(dataset);
      println(arr)
    }

  }
}
