package datapipelines;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.canova.common.RecordConverter;
import org.canova.sound.recordreader.WavFileRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

import java.io.File;
import java.util.Collection;


public class AudioVectorizer {
    public static void main( String[] args )  throws Exception {

        String filepath = System.getProperty("user.home")
                +"/dsr/data/mlsp_contest_dataset/essential_data/src_wavs";

        RecordReader wavRecordReader = new WavFileRecordReader(true);
        wavRecordReader.initialize(new FileSplit(new File(filepath)));

        DataSetIterator iter = new RecordReaderDataSetIterator(wavRecordReader);

        while(iter.hasNext()) {
           DataSet next = iter.next();
           //network.fit(next);
       }

    }

}