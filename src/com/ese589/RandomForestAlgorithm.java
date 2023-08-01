package com.ese589;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class RandomForestAlgorithm {
	int numberOfTrees;
	List<String> outputList = new ArrayList<>();
	HashMap<Integer, ArrayList<String>> listOfAllDiscreteValuesBasedOnAttributeNumber;
	String output1, output2;
	ArrayList<String[]> trainingDataSetAfterPreProcessing;
	ArrayList<String[]> testingDataSetAfterPreProcessing;
	double trainingTimeForExcecution = 0;
	double precisionValue;
	double recallValue;
	double fscoreValue;
	double accuracyValue;
	DecisionTreeServiceImpl[] decisionTree;

	public RandomForestAlgorithm(int numberOfTrees, double attributesFraction, double trainingFraction,
			ArrayList<String[]> trainingDataSetAfterPreProcessing, ArrayList<String[]> testingDataSetAfterPreProcessing,
			List<Integer> numberOfTotalOutputs, List<String> outputList,
			HashMap<Integer, ArrayList<String>> listOfAllDiscreteValuesBasedOnAttributeNumber) {

		this.numberOfTrees = numberOfTrees;
		this.trainingDataSetAfterPreProcessing = trainingDataSetAfterPreProcessing;
		this.output1 = outputList.get(0);
		this.output2 = outputList.get(1);
		this.testingDataSetAfterPreProcessing = testingDataSetAfterPreProcessing;
		this.listOfAllDiscreteValuesBasedOnAttributeNumber = listOfAllDiscreteValuesBasedOnAttributeNumber;

		int numberOfAttributes = trainingDataSetAfterPreProcessing.get(0).length - 1,
				numberOfTrainingInstances = trainingDataSetAfterPreProcessing.size();
		int numberOfRandomInstances = (int) (numberOfTrainingInstances * trainingFraction);
		int numberOfRandomAttributes = (int) (numberOfAttributes * attributesFraction);

		String[][] transactionDataSetInArray = new String[numberOfTrainingInstances][];
		int q = 0;
		for (String[] transaction : trainingDataSetAfterPreProcessing) {
			transactionDataSetInArray[q++] = transaction;
		}

		trainingTimeForExcecution = System.currentTimeMillis();

		Random random = new Random();
		decisionTree = new DecisionTreeServiceImpl[numberOfTrees];
		for (int i = 0; i < numberOfTrees; i++) {
			ArrayList<Integer> remainingAttrributes = new ArrayList<>();
			for (int j = 0; j < numberOfRandomAttributes; j++) {
				int ran = random.nextInt(numberOfAttributes);
				if (remainingAttrributes.contains(ran))
					j--;
				else
					remainingAttrributes.add(ran);
			}
			ArrayList<String[]> randomDataSet = new ArrayList<>();
			for (int j = 0; j < numberOfRandomInstances; j++) {
				randomDataSet.add(transactionDataSetInArray[random.nextInt(numberOfTrainingInstances)]);
			}
			decisionTree[i] = new DecisionTreeServiceImpl(randomDataSet, testingDataSetAfterPreProcessing,
					numberOfTotalOutputs, outputList, listOfAllDiscreteValuesBasedOnAttributeNumber,
					remainingAttrributes);
		}

		trainingTimeForExcecution = (System.currentTimeMillis() - trainingTimeForExcecution) / 1000.0;

		randomForestAnalysis();
	}

	public void randomForestAnalysis() {
		if (decisionTree == null)
			return;
		int correctClassificationOutput = 0, incorrectClassificationOutput = 0;
		int tf_TruePositive = 0, fp_FalsePositive = 0, fn_FalseNegative = 0;
		for (String[] s : testingDataSetAfterPreProcessing) {
			int temp1 = 0, temp2 = 0;
			for (int i = 0; i < numberOfTrees; i++) {
				if (DecisionTreeNode.predictionOfOutput(decisionTree[i].decisionTreeNode, s,
						listOfAllDiscreteValuesBasedOnAttributeNumber) == 1)
					temp1++;
				else
					temp2++;
			}
			int predictedValue = temp1 >= temp2 ? 1 : 2,
					actualValue = s[s.length - 1].equals(output1) ? 1 : 2;
			if (predictedValue == actualValue)
				correctClassificationOutput++;
			else
				incorrectClassificationOutput++;
			if (predictedValue == 1 && actualValue == 1)
				tf_TruePositive++;
			else if (predictedValue == 1 && actualValue == 2)
				fn_FalseNegative++;
			else if (predictedValue == 2 && actualValue == 1)
				fp_FalsePositive++;
		}
		precisionValue = tf_TruePositive / (tf_TruePositive + fp_FalsePositive + 0.0);
		recallValue = tf_TruePositive / (tf_TruePositive + fn_FalseNegative + 0.0);
		fscoreValue = 2 * precisionValue * recallValue / (precisionValue + recallValue);
		accuracyValue = (correctClassificationOutput) / (correctClassificationOutput + incorrectClassificationOutput + 0.0);
		System.out.println("Training Time in secs = " + trainingTimeForExcecution);
		System.out.println("Accuracy=" + accuracyValue + "\nPrecision=" + precisionValue + "\nRecall=" + recallValue
				+ "\nF-Score=" + fscoreValue);
	}
}
