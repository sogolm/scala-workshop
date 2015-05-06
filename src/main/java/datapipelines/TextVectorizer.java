package datapipelines;

import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CollectionRecordReader;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.canova.api.writable.Writables;
import org.canova.nd4j.nlp.vectorizer.TfidfVectorizer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

/**
 * Created by sogolmoshtaghi on 4/27/15.
 */
public class TextVectorizer {
    public static void main(String[] args) throws Exception{
        String filepath = System.getProperty("user.home")
                +"/dsr/data/SentenceCorpus";

        RecordReader recordReader = new FileRecordReader();
        recordReader.initialize(new FileSplit(new File(filepath)));
        TfidfVectorizer vectorizer = new org.canova.nd4j.nlp.vectorizer.TfidfVectorizer();
        vectorizer.initialize(new Configuration());
        INDArray n = vectorizer.fitTransform(recordReader);
        System.out.println(n.columns());
        System.out.println(n.rows());

        LayerFactory factory = LayerFactories.getFactory(OutputLayer.class);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().layerFactory(factory)
                .activationFunction("softmax").weightInit(WeightInit.ZERO)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .nIn(n.columns()).nOut(3).build();
        OutputLayer logistic = factory.create(conf);
        logistic.fit(n);
        //output is for new examples
        Evaluation eval = new Evaluation();
        //pass in to eval.eval true labels and guesses from logistic.output

    }
}
