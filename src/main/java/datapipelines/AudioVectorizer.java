package datapipelines;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.canova.common.RecordConverter;
import org.canova.sound.recordreader.WavFileRecordReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.io.File;
import java.util.Collection;

/**
 * Hello world!
 *
 */
public class AudioVectorizer {
    public static void main( String[] args )  throws Exception {

        String filepath = System.getProperty("user.home")
                +"/dsr/data/mlsp_contest_dataset/essential_data/src_wavs";

        RecordReader wavRecordReader = new WavFileRecordReader(true);
        wavRecordReader.initialize(new FileSplit(new File(filepath)));
        while(wavRecordReader.hasNext()) {
           Collection<Writable> dataset = wavRecordReader.next();
           INDArray arr = RecordConverter.toArray(dataset);
           System.out.println(arr);
       }

    }

}

