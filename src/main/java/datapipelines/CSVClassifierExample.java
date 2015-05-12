package datapipelines;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.StringEscapeUtils;
import org.apache.hadoop.io.Writable;
import org.canova.api.io.WritableConverter;
import org.canova.api.io.converters.WritableConverterException;
import org.canova.api.io.data.IntWritable;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.common.RecordConverter;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.CSVDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;


public class CSVClassifierExample {

    private static final Logger log = LoggerFactory.getLogger(CSVClassifierExample.class);

    public static void main(String[] args) throws Exception {

        // Path to the csv file
        String path = System.getProperty("user.home") + "/dsr/data/IRIS_dataset/iris.txt";

        //Instantiating a RecordReader pointing to the data path
        RecordReader reader = new CSVRecordReader();
        reader.initialize(new FileSplit(new File(path)));

        final List<String> labels = Arrays.asList("Iris-setosa","Iris-versicolor","Iris-virginica");


        //Instantiating a DataSetIterator to traverse through the data
        // with the given batch size, label column index, and possible number of labels.
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(reader,new WritableConverter() {
            @Override
            public org.canova.api.writable.Writable convert(org.canova.api.writable.Writable writable) throws WritableConverterException {
                if(writable instanceof Text) {
                    String s = writable.toString().replaceAll("[\u0000]+", "");
                    int idx = labels.indexOf(s);
                    return new IntWritable(idx);
                }
                return writable;
            }
        } ,50, -1, 3);

        //Building a classifier -- simple logistic regression
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .activationFunction("softmax")
                .iterations(100)
                .weightInit(WeightInit.ZERO)
                .learningRate(1e-1)
                .nIn(4)
                .nOut(3)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer())
                .build();
        org.deeplearning4j.nn.layers.OutputLayer classifier = LayerFactories.getFactory(conf.getLayer())
                .create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)));

        // Iterating over the data and training the model with each batch.
        while (iter.hasNext()) {
            DataSet next =  iter.next();
            classifier.fit(next);
        }


        //Classifying test data using the model -- I'm using the same train data as test in this case.
        iter.reset(); //TODO: should we do this in a loop?
        while(iter.hasNext()){
            DataSet test = iter.next();
            test.normalizeZeroMeanZeroUnitVariance();
            Evaluation eval = new Evaluation();
            INDArray output = classifier.output(test.getFeatureMatrix()); //NdArray of likelihood probabilities for each row
            eval.eval(test.getLabels(), output);
            System.out.println("Score " + eval.stats());
        }
    }
}
