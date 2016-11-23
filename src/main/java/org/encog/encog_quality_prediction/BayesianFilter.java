package org.encog.encog_quality_prediction;

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

	private int k;
	
	private BagOfWords goodBag;
	private BagOfWords badBag;
	private BagOfWords totalBag;
	
	public void init(int theK) {
		
		this.k = theK;
		
		this.goodBag = new BagOfWords(this.k);
		this.badBag = new BagOfWords(this.k);
		this.totalBag = new BagOfWords(this.k);

		
		for(String line: GOOD_DATA) {
			goodBag.process(line);
			totalBag.process(line);
		}
		
		for(String line: BAD_DATA) {
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
		messageProbability.addClass(GOOD_DATA.length);
		messageProbability.addClass(BAD_DATA.length);
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
	
	private static final String FILENAME = "bayes_test.csv";
	
	public void test(String message) {
		double d = probabilityGood(message);
		System.out.println("Probability of image with caption \"" + message + "\" being good is " + Format.formatPercent(d));
	}
	
	public static void main(String args[]){
		//final MLDataSet trainingSet = TrainingSetUtil.loadCSVTOMemory(CSVFormat.ENGLISH, FILENAME, true, 1, 1);
		
		
		BayesianFilter program = new BayesianFilter();
		
		System.out.println("Using Laplace of 0");
		program.init(0);
		for(int i=1;i<=12;i++){
			program.test(""+i);
		}
		
		System.out.println("Using Laplace of 1");
		program.init(1);
		for(int i=1;i<=12;i++){
			program.test(""+i);
		}
	}
}
