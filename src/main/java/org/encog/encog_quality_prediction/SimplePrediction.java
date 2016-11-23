package org.encog.encog_quality_prediction;

import org.encog.ml.data.MLDataSet;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;
import org.encog.util.simple.TrainingSetUtil;

public class SimplePrediction {

	private static final String FILENAME = "training.csv";
	
	public static void main(String args[]){
		final MLDataSet trainingSet = TrainingSetUtil.loadCSVTOMemory(CSVFormat.ENGLISH, FILENAME, true, 5, 1);
		
		//final BasicNetwork network = EncogUtility.simpleFeedForward(input, hidden1, hidden2, output, tanh);
	}
}
