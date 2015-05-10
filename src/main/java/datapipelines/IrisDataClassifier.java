package datapipelines;

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
import java.util.Iterator;

/**
 * Created by sogolmoshtaghi on 5/8/15.
 */
public class IrisDataClassifier {
    private static final Logger log = LoggerFactory.getLogger(IrisDataClassifier.class);

    public static void main(String[] args) throws Exception{

        log.info("Path to the csv file.");
        String dataPath = System.getProperty("user.home")
                +"/dsr/data/IRIS_dataset/iris.txt";

        log.info("Instantiating a DataSetIterator to traverse the data set with the given batch size.");
        DataSetIterator iter = new CSVDataSetIterator(150, 150, new File(dataPath), 4);

        log.info("Building classifier.");
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT).optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .activationFunction("softmax")
                .iterations(100).weightInit(WeightInit.ZERO)
                .learningRate(1e-1).nIn(4).nOut(3).layer(new org.deeplearning4j.nn.conf.layers.OutputLayer()).build();
        org.deeplearning4j.nn.layers.OutputLayer classifier = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)));

        log.info("Iterating over the data and training the model with each batch.");
        ArrayList<DataSet> testSet = new ArrayList<DataSet>();
        while (iter.hasNext()) {
            DataSet dataset = iter.next();
            int trainSize = (int) (dataset.getFeatureMatrix().size(0) * 0.8);
            SplitTestAndTrain trainTest = dataset.splitTestAndTrain(trainSize);
            trainTest.getTrain().normalizeZeroMeanZeroUnitVariance();
            classifier.fit(trainTest.getTrain().getFeatureMatrix());
            testSet.add(trainTest.getTest());
        }

        log.info("Classifying test data using the model.");
        Iterator<DataSet> testIter = testSet.iterator();
        while(testIter.hasNext()){
            DataSet test = testIter.next();
            test.normalizeZeroMeanZeroUnitVariance();
            Evaluation eval = new Evaluation();
            INDArray output = classifier.output(test.getFeatureMatrix()); //NdArray of likelihood probabilities for each row
            eval.eval(test.getLabels(), output);
            log.info("Score " + eval.stats());
        }
    }
}
