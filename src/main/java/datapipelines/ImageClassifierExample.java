package datapipelines;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.slf4j.LoggerFactory;


public class ImageClassifierExample {

    public static void main(String[] args) throws IOException, InterruptedException {


        // Path to the labeled images
        String labeledPath = System.getProperty("user.home")+"/lfw";
         List<String> labels = new ArrayList<>();
        for(File f : new File(labeledPath).listFiles()) {
            labels.add(f.getName());
        }
        // Instantiating a RecordReader pointing to the data path with the specified
        // height and width for each image.
        RecordReader recordReader = new ImageRecordReader(28, 28, true,labels);
        recordReader.initialize(new FileSplit(new File(labeledPath)));

        // Canova to Dl4j
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 784,labels.size());

        // Creating configuration for the neural net.
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .constrainGradientToUnitNorm(true)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(1,1e-5))
                .iterations(100).learningRate(1e-3)
                .nIn(784).nOut(labels.size())
                .visibleUnit(org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit.RECTIFIED)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM())
                .list(4).hiddenLayerSizes(600, 250, 100).override(3, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        if (i == 3) {
                            builder.layer(new org.deeplearning4j.nn.conf.layers.OutputLayer());
                            builder.activationFunction("softmax");
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);

                        }
                    }
                }).build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.setListeners(Arrays.<IterationListener>asList(new ScoreIterationListener(10)));

        // Training
        while(iter.hasNext()){
            DataSet next = iter.next();
            network.fit(next);
        }

        // Testing -- We're not doing split test and train
        // Using the same training data as test.
        iter.reset();
        Evaluation eval = new Evaluation();
        while(iter.hasNext()){
            DataSet next = iter.next();
            INDArray predict2 = network.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), predict2);
        }

        System.out.println(eval.stats());
    }
}
