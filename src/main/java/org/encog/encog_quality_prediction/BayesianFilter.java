package org.encog.encog_quality_prediction;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.encog.mathutil.probability.CalcProbability;
import org.encog.ml.bayesian.BayesianEvent;
import org.encog.ml.bayesian.BayesianNetwork;
import org.encog.ml.bayesian.EventType;
import org.encog.ml.bayesian.query.enumerate.EnumerationQuery;
import org.encog.ml.data.MLDataSet;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.TrainingSetUtil;
import org.encog.util.text.BagOfWords;

public class BayesianFilter {
	
	public final static String[] GOOD_DATA = {
			"1 2 3",
			"4 3 5",
			"3 6 5"
	};
	
	public final static String[] BAD_DATA = {
			"7 6 8",
			"10 7 6",
			"3 6 9",
			"6 2 8",
			"6 11 12"
	};
	
	public List<String> goodImageName;
	
	public List<String> badImageName;

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
		goodImageName = new ArrayList<String>();
		badImageName = new ArrayList<String>();
		try{
			reader = new BufferedReader(new FileReader(filename));
			reader.readLine();//skip header
			while((line = reader.readLine())!=null){
				String[] entry = line.split(separator);
		
				if("1".equals(entry[9])){
					if(!entry[6].isEmpty()) goodImageName.add(entry[6]);
				}else{
					if(!entry[6].isEmpty()) badImageName.add(entry[6]);
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

		
		for(String line: goodImageName) {
			goodBag.process(line);
			totalBag.process(line);
		}
		
		for(String line: badImageName) {
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
		
		//SamplingQuery query = new SamplingQuery(network);
		EnumerationQuery query = new EnumerationQuery(network);
		
		CalcProbability messageProbability = new CalcProbability(this.k);
		messageProbability.addClass(goodImageName.size());
		messageProbability.addClass(badImageName.size());
		double probSpam = messageProbability.calculate(0);

		goodEvent.getTable().addLine(probSpam, true);
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
		
		System.out.println("Using Laplace of 0");
		program.init(0);
		program.total = 0;
		program.accurate = 0;
		try{
			reader = new BufferedReader(new FileReader(TESTING_FILE));
			reader.readLine();//skip header
			while((line = reader.readLine())!=null){
				String[] entry = line.split(separator);
				program.total++;
				program.test(entry[6],"1".equals(entry[9]));
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		System.out.println(program.accurate + "/" + program.total);
		
		System.out.println("Using Laplace of 1");
		program.init(1);
		program.total = 0;
		program.accurate = 0;
		try{
			reader = new BufferedReader(new FileReader(TESTING_FILE));
			reader.readLine();//skip header
			while((line = reader.readLine())!=null){
				String[] entry = line.split(separator);
				program.total++;
				program.test(entry[6],"1".equals(entry[9]));
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		System.out.println(program.accurate + "/" + program.total);
		
	}
}
