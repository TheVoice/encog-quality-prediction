package org.encog.encog_quality_prediction;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

/**
 * XOR: This example is essentially the "Hello World" of neural network
 * programming.  This example shows how to construct an Encog neural
 * network to predict the output from the XOR operator.  This example
 * uses backpropagation to train the neural network.
 * 
 * This example attempts to use a minimum of Encog features to create and
 * train the neural network.  This allows you to see exactly what is going
 * on.  For a more advanced example, that uses Encog factories, refer to
 * the XORFactory example.
 * 
 */
public class XORBased {

	private static final String TRAINING_FILE = "training.csv";
	private static final int TRAINING_SET_SIZE = 1000;
	/**
	 * The input necessary for XOR.
	 */
	public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
			{ 0.0, 1.0 }, { 1.0, 1.0 } };

	/**
	 * The ideal data necessary for XOR.
	 */
	public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };
	
	/**
	 * The main method.
	 * @param args No arguments are used.
	 */
	public static void main(final String args[]) {
		
		// create a neural network, without using a factory
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null,true,2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(),true,3));
		network.addLayer(new BasicLayer(new ActivationSigmoid(),false,1));
		network.getStructure().finalizeStructure();
		network.reset();
		
		//read training file
		BufferedReader reader = null;
		String line = "";
		String separator = ",";
		ArrayList<ArrayList<Double>> quality_input = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> quality_ideal = new ArrayList<ArrayList<Double>>();
		try{
			reader = new BufferedReader(new FileReader(TRAINING_FILE));
			line = reader.readLine();//discard first line
			for(int i=0;i<TRAINING_SET_SIZE;i++){
				line = reader.readLine();
				ArrayList<Double> singleInput = new ArrayList<Double>();
				String[] splittedLine = line.split(separator);
				singleInput.add(Double.parseDouble(splittedLine[1]));
				singleInput.add(Double.parseDouble(splittedLine[2]));
				singleInput.add(Double.parseDouble(splittedLine[3]));
				singleInput.add(Double.parseDouble(splittedLine[4]));
				singleInput.add(Double.parseDouble(splittedLine[5]));
				//TODO resolve name,description,caption handling
				
				quality_input.add(singleInput);
				
				ArrayList<Double> singleOutput = new ArrayList<Double>();
				singleOutput.add(Double.parseDouble(splittedLine[9]));
				quality_ideal.add(singleOutput);
			}
		}catch(Exception e){
			e.printStackTrace();
		}finally{
			if(reader != null){
				try{
					reader.close();
				}catch (IOException e){
					e.printStackTrace();
				}
			}
		}

		System.out.println(quality_input.get(1));
		System.out.println(quality_ideal.get(1));

		// create training data
		double[][] quality_input_array = new double[quality_input.size()][];
		for(int i = 0;i<quality_input.size();i++){
			ArrayList<Double> row = quality_input.get(i);
			double[] target = new double[row.size()];
			Iterator<Double> iterator = row.iterator();
			int j = 0;
			while(iterator.hasNext()){
				target[j] = iterator.next();
				j++;
			}
			quality_input_array[i] = target;
		}
		
		double[][] quality_ideal_array = new double[quality_ideal.size()][];
		for(int i = 0;i<quality_ideal.size();i++){
			ArrayList<Double> row = quality_ideal.get(i);
			double[] target = new double[row.size()];
			Iterator<Double> iterator = row.iterator();
			int j = 0;
			while(iterator.hasNext()){
				target[j] = iterator.next();
				j++;
			}
			quality_ideal_array[i] = target;
		}
		MLDataSet trainingSet = new BasicMLDataSet(quality_input_array, quality_ideal_array);
		
		// train the neural network
		final MLTrain train = new ResilientPropagation(network, trainingSet);

		int epoch = 1;

		do {
			train.iteration();
			System.out.println("Epoch #" + epoch + " Error:" + train.getError());
			epoch++;
		} while(train.getError() > 0.1);
		train.finishTraining();

		// test the neural network
		System.out.println("Neural Network Results:");
		for(MLDataPair pair: trainingSet ) {
			final MLData output = network.compute(pair.getInput());
			System.out.println(pair.getInput().getData(0) + "," + pair.getInput().getData(1) + "," + pair.getInput().getData(2) + "," + pair.getInput().getData(3) + "," + pair.getInput().getData(4)
					+ ", actual=" + output.getData(0) + ",ideal=" + pair.getIdeal().getData(0));
		}
		
		Encog.getInstance().shutdown();
	}
}
