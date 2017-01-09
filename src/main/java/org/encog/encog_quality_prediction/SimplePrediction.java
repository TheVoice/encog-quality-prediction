
package org.encog.encog_quality_prediction;

import java.io.File;
import java.util.Arrays;

import org.encog.ConsoleStatusReportable;
import org.encog.Encog;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.versatile.NormalizationHelper;
import org.encog.ml.data.versatile.VersatileMLDataSet;
import org.encog.ml.data.versatile.columns.ColumnDefinition;
import org.encog.ml.data.versatile.columns.ColumnType;
import org.encog.ml.data.versatile.sources.CSVDataSource;
import org.encog.ml.data.versatile.sources.VersatileDataSource;
import org.encog.ml.factory.MLMethodFactory;
import org.encog.ml.model.EncogModel;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;
import org.encog.util.simple.EncogUtility;

public class SimplePrediction {

  private static final String FILENAME = "train.csv";

  public static void main(String args[]) {
    // final MLDataSet trainingSet = TrainingSetUtil.loadCSVTOMemory(CSVFormat.ENGLISH, FILENAME, true, 5, 1);
    // final BasicNetwork network = EncogUtility.simpleFeedForward();

    try {
      File trainFile = new File(FILENAME);

      // Define the format of the data file.
      // This area will change, depending on the columns and
      // format of the file that you are trying to model.
      VersatileDataSource source = new CSVDataSource(trainFile, false, CSVFormat.DECIMAL_POINT);
      VersatileMLDataSet data = new VersatileMLDataSet(source);
      data.defineSourceColumn("latitude", 1, ColumnType.continuous);
      data.defineSourceColumn("longtitude", 2, ColumnType.continuous);
      data.defineSourceColumn("width", 3, ColumnType.continuous);
      data.defineSourceColumn("height", 4, ColumnType.continuous);
      data.defineSourceColumn("size", 5, ColumnType.continuous);
      data.defineSourceColumn("name", 6, ColumnType.continuous);
      data.defineSourceColumn("description", 7, ColumnType.continuous);

      // Define the column that we are trying to predict.
      ColumnDefinition outputColumn = data.defineSourceColumn("good", 8, ColumnType.nominal);

      // Analyze the data, determine the min/max/mean/sd of every column.
      data.analyze();

      // Map the prediction column to the output of the model, and all
      // other columns to the input.
      data.defineSingleOutputOthersInput(outputColumn);

      // Create feedforward neural network as the model type. MLMethodFactory.TYPE_FEEDFORWARD.
      // You could also other model types, such as:
      // MLMethodFactory.SVM: Support Vector Machine (SVM)
      // MLMethodFactory.TYPE_RBFNETWORK: RBF Neural Network
      // MLMethodFactor.TYPE_NEAT: NEAT Neural Network
      // MLMethodFactor.TYPE_PNN: Probabilistic Neural Network
      EncogModel model = new EncogModel(data);
      model.selectMethod(data, MLMethodFactory.TYPE_NEAT);

      // Send any output to the console.
      model.setReport(new ConsoleStatusReportable());

      // Now normalize the data. Encog will automatically determine the correct normalization
      // type based on the model you chose in the last step.
      data.normalize();

      // Hold back some data for a final validation.
      // Shuffle the data into a random ordering.
      // Use a seed of 1001 so that we always use the same holdback and will get more consistent results.
      model.holdBackValidation(0.3, true, 1001);

      // Choose whatever is the default training type for this model.
      model.selectTrainingType(data);

      // Use a 5-fold cross-validated train. Return the best method found.
      MLRegression bestMethod = (MLRegression) model.crossvalidate(5, true);

      // Display the training and validation errors.
      System.out.println("Training error: " + EncogUtility.calculateRegressionError(bestMethod, model.getTrainingDataset()));
      System.out.println("Validation error: " + EncogUtility.calculateRegressionError(bestMethod, model.getValidationDataset()));

      // Display our normalization parameters.
      NormalizationHelper helper = data.getNormHelper();
      System.out.println(helper.toString());

      // Display the final model.
      System.out.println("Final model: " + bestMethod);

      // Loop over the entire, original, dataset and feed it through the model.
      // This also shows how you would process new data, that was not part of your
      // training set. You do not need to retrain, simply use the NormalizationHelper
      // class. After you train, you can save the NormalizationHelper to later
      // normalize and denormalize your data.
      ReadCSV csv = new ReadCSV(trainFile, false, CSVFormat.DECIMAL_POINT);
      String[] line = new String[7];
      MLData input = helper.allocateInputVector();

      while (csv.next()) {
        StringBuilder result = new StringBuilder();
        line[0] = csv.get(1);
        line[1] = csv.get(2);
        line[2] = csv.get(3);
        line[3] = csv.get(4);
        line[4] = csv.get(5);
        line[5] = csv.get(6);
        line[6] = csv.get(7);
        
        String correct = csv.get(8);
        helper.normalizeInputVector(line, input.getData(), false);
        MLData output = bestMethod.compute(input);
        String qualityChosen = helper.denormalizeOutputVectorToString(output)[0];

        result.append(Arrays.toString(line));
        result.append(" -> predicted: ");
        result.append(qualityChosen);
        result.append(" (correct: ");
        result.append(correct);
        result.append(")");
        if (qualityChosen.equals(correct)) {
          result.append(" + ");
        } else {
          result.append(" - ");
        }

        System.out.println(result.toString());
      }

      // Delete data file and shut down.
      trainFile.delete();
      Encog.getInstance().shutdown();

    } catch (Exception ex) {
      ex.printStackTrace();
    }

  }
}
