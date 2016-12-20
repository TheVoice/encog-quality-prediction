package org.encog.encog_quality_prediction;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.encog.mathutil.probability.CalcProbability;
import org.encog.ml.bayesian.BayesianEvent;
import org.encog.ml.bayesian.BayesianNetwork;
import org.encog.ml.bayesian.EventType;
import org.encog.ml.bayesian.query.enumerate.EnumerationQuery;
import org.encog.ml.bayesian.query.sample.SamplingQuery;
import org.encog.ml.data.MLDataSet;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.TrainingSetUtil;
import org.encog.util.text.BagOfWords;

public class BayesianFilter {

	private static final int LAPLACE = 1;

	private static final boolean TAKE_AVERAGE = true;

	public List<String> goodImageTags;

	public List<String> badImageTags;

	private int k;

	private BagOfWords goodBag;
	private BagOfWords badBag;
	private BagOfWords totalBag;

	public int accurate;
	public int total;

	public void loadCsvFile(String filename){
		BufferedReader reader;
		String line;
		String separator = ",";
		goodImageTags = new ArrayList<String>();
		badImageTags = new ArrayList<String>();
		try{
			reader = new BufferedReader(new FileReader(filename));
			reader.readLine();//skip header
			while((line = reader.readLine())!=null){
				String[] entry = line.split(separator);

				if("1".equals(entry[9])){
					if(!entry[6].isEmpty()) goodImageTags.add(entry[6]);
					if(!entry[7].isEmpty()) goodImageTags.add(entry[7]);
					if(!entry[8].isEmpty()) goodImageTags.add(entry[8]);
				}else{
					if(!entry[6].isEmpty()) badImageTags.add(entry[6]);
					if(!entry[7].isEmpty()) badImageTags.add(entry[7]);
					if(!entry[8].isEmpty()) badImageTags.add(entry[8]);
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
	}

	public void init(int theK) {

		this.k = theK;

		this.goodBag = new BagOfWords(this.k);
		this.badBag = new BagOfWords(this.k);
		this.totalBag = new BagOfWords(this.k);


		for(String line: goodImageTags) {
			goodBag.process(line);
			totalBag.process(line);
		}

		for(String line: badImageTags) {
			badBag.process(line);
			totalBag.process(line);
		}

		this.badBag.setLaplaceClasses(totalBag.getUniqueWords());
		this.goodBag.setLaplaceClasses(totalBag.getUniqueWords());		
	}

	//funkcja do rozdzielania słów
	public List<String> separateSpaces(String str) {
		List<String> result = new ArrayList<String>();
		StringBuilder word = new StringBuilder();

		for (int i = 0; i < str.length(); i++) {
			char ch = str.charAt(i);
			if (ch != '\'' && !Character.isLetterOrDigit(ch)) {
				if (word.length() > 0) {
					result.add(word.toString());
					word.setLength(0);
				}
			} else {
				word.append(ch);
			}
		}

		if (word.length() > 0) {
			result.add(word.toString());
		}

		return result;
	}

	public double probabilityGood(String m) {
		List<String> words = separateSpaces(m);

		BayesianNetwork network = new BayesianNetwork();
		BayesianEvent goodEvent = network.createEvent("good");

		int index = 0;
		for( String word: words) {
			BayesianEvent event = network.createEvent(word+index);
			network.createDependency(goodEvent, event);
			index++;
		}

		network.finalizeStructure();

		//SamplingQuery is too complicated for our example
		//SamplingQuery query = new SamplingQuery(network);
		EnumerationQuery query = new EnumerationQuery(network);

		CalcProbability messageProbability = new CalcProbability(this.k);
		messageProbability.addClass(goodImageTags.size());
		messageProbability.addClass(badImageTags.size());
		double probGood = messageProbability.calculate(0);

		goodEvent.getTable().addLine(probGood, true);
		query.defineEventType(goodEvent, EventType.Outcome);
		query.setEventValue(goodEvent, true);

		index = 0;
		for( String word: words) {
			String word2 = word+index;
			BayesianEvent event = network.getEvent(word2);
			event.getTable().addLine(this.goodBag.probability(word), true, true); // good
			event.getTable().addLine(this.badBag.probability(word), true, false); // bad
			query.defineEventType(event, EventType.Evidence);
			query.setEventValue(event, true);
			index++;
		}

		//query.setSampleSize(100000000);
		query.execute();
		return query.getProbability();		
	}

	private static final String TRAINING_FILE = "training_without_test.csv";
	private static final String TESTING_FILE = "bayes_test.csv";

	public void test(String message,boolean actualGood) {
		double d = probabilityGood(message);
		System.out.print("Probability of image with name \"" + message + "\" being good is " + Format.formatPercent(d));
		if(d>=50.0){
			if(actualGood){
				System.out.println(" CORRECT");
				this.accurate++;
			}else{
				System.out.println(" INCORRECT");
			}
		}else{
			if(actualGood){
				System.out.println(" INCORRECT");
			}else{
				System.out.println(" CORRECT");
				this.accurate++;
			}
		}
	}



	public static void main(String args[]){
		//final MLDataSet trainingSet = TrainingSetUtil.loadCSVTOMemory(CSVFormat.ENGLISH, FILENAME, true, 1, 1);


		BayesianFilter program = new BayesianFilter();
		program.loadCsvFile(TRAINING_FILE);

		BufferedReader reader;
		String separator = ",";
		String line;
		PrintWriter writer = null;
		Date date = new Date();
		SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");

		System.out.println("Using Laplace of "+LAPLACE);
		program.init(LAPLACE);
		program.total = 0;
		program.accurate = 0;
		try{
			writer = new PrintWriter("bayes_filtered_"+dateFormat.format(date)+".csv");
			reader = new BufferedReader(new FileReader(TESTING_FILE));
			reader.readLine();//skip header
			while((line = reader.readLine())!=null){
				String[] entry = line.split(separator);
				program.total++;
				//program.test(entry[6],"1".equals(entry[9]));
				program.writeLineToOutput(entry,writer);
			}
		}catch(Exception e){
			e.printStackTrace();
		}finally{
			writer.close();
		}
		System.out.println("DONE");
		//System.out.println(program.accurate + "/" + program.total);

	}

	private void writeLineToOutput(String[] entry, PrintWriter writer) {
		writer.print(entry[0]);
		writer.print(",");
		writer.print(entry[1]);
		writer.print(",");
		writer.print(entry[2]);
		writer.print(",");
		writer.print(entry[3]);
		writer.print(",");
		writer.print(entry[4]);
		writer.print(",");
		writer.print(entry[5]);
		writer.print(",");
		double nameProb = probabilityGood(entry[6]);
		double descriptionProb = probabilityGood(entry[7]);
		double captionProb = probabilityGood(entry[8]);
		if(TAKE_AVERAGE){
			double sum = 0.0;
			double elements = 0.0;
			if(!Double.isNaN(nameProb)){
				sum += nameProb;
				elements += 1.0;
			}
			if(!Double.isNaN(descriptionProb)){
				sum += descriptionProb;
				elements += 1.0;
			}
			if(!Double.isNaN(captionProb)){
				sum += captionProb;
				elements += 1.0;
			}
			if(elements!=0.0)
				writer.print(String.format("%.8f", sum / elements));
		}else{
			String nameProbStr = (!Double.isNaN(nameProb)) ? String.format("%.8f", nameProb) : "";
			writer.print(nameProbStr);
			writer.print(",");
			String descriptionProbStr = (!Double.isNaN(descriptionProb)) ? String.format("%.8f", descriptionProb) : "";
			writer.print(descriptionProbStr);
			writer.print(",");
			String captionProbStr = (!Double.isNaN(captionProb)) ? String.format("%.8f", captionProb) : "";
			writer.print(captionProbStr);
		}
		writer.print(",");
		writer.print(entry[9]);
		writer.println();
	}
}
