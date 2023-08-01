package com.ese589;

import java.util.ArrayList;
import java.util.HashMap;
import java.io.IOException;
import java.io.File;
import java.util.List;

public class DecisionTreeMain {

	public static void main(String[] args) throws IOException {

		List<String> outputList = new ArrayList<>();
		// We need to enter the different outputs of the supervised data sets
		String output1 = "<=50K";
		String output2 = ">50K";
		outputList.add(output1);
		outputList.add(output2);
		List<Integer> columnNumberForContinuousAttributesToMakeItDiscrete = new ArrayList<>();
		// We need to enter the different column number of the continuous values to make
		// it discrete
		// for example, salary of people, people ages as it is different for each
		// transaction so we make it discrete using minimum Entropy
		columnNumberForContinuousAttributesToMakeItDiscrete.add(0);
		columnNumberForContinuousAttributesToMakeItDiscrete.add(2);
		columnNumberForContinuousAttributesToMakeItDiscrete.add(4);
		columnNumberForContinuousAttributesToMakeItDiscrete.add(10);
		columnNumberForContinuousAttributesToMakeItDiscrete.add(11);
		columnNumberForContinuousAttributesToMakeItDiscrete.add(12);
		DecisionTreePreProcessing preProcessing = new DecisionTreePreProcessing(outputList,
				columnNumberForContinuousAttributesToMakeItDiscrete);
		ArrayList<String[]> trainingDataSetAfterPreProcessing = preProcessing
				.preProcessingDataSetToMakeItDiscrete(new File("train.data"));
		List<Integer> numberOfTotalOutputs = preProcessing.numberOfOutputs;
		DecisionTreePreProcessing.predictMissingOrEmptyValues(trainingDataSetAfterPreProcessing, "train");
		DecisionTreePreProcessing.computelistOfAllDiscreteValuesForEachColumn(trainingDataSetAfterPreProcessing);
		ArrayList<String[]> testDataAfterPreProcessing = DecisionTreePreProcessing
				.preProcessingDataSetToMakeItDiscreteTestData(new File("test.data"),
						columnNumberForContinuousAttributesToMakeItDiscrete);
		ArrayList<Integer> remainingAttributesToBeAddedInTree = new ArrayList<>();
		for (int i = 0; i < trainingDataSetAfterPreProcessing.get(0).length - 1; i++) {
			remainingAttributesToBeAddedInTree.add(i);
		}
		HashMap<Integer, ArrayList<String>> listOfAllDiscreteValuesBasedOnAttributeNumber = DecisionTreePreProcessing.listOfAllDiscreteValuesBasedOnAttributeNumber;

		System.out.println("Generating Decision Tree");
		DecisionTreeServiceImpl decisionTree = new DecisionTreeServiceImpl(trainingDataSetAfterPreProcessing,
				testDataAfterPreProcessing, numberOfTotalOutputs, outputList,
				listOfAllDiscreteValuesBasedOnAttributeNumber, remainingAttributesToBeAddedInTree);
		decisionTree.printExecutionTimeOfDecisionTree();
		decisionTree.printAnalysisofdecisionTree();

		System.out.println("\nReduced Error Pruning");

		ReducedErrorPruningOnDecisionTree reducedErrorPruningOnDecisionTree = new ReducedErrorPruningOnDecisionTree(
				decisionTree);
		reducedErrorPruningOnDecisionTree.decisionTree.printAnalysisofdecisionTree();

		int numberOftree = 10;
		double attributesFraction = 0.5;
		double trainingFraction = 0.33;
		System.out.println("\nRandom Forest");
		
		System.out.println( numberOftree + " trees, " + attributesFraction
				+ " attributes fraction, " + trainingFraction + " training instances fraction in each tree");
		RandomForestAlgorithm randomForestAlgorithm = new RandomForestAlgorithm(numberOftree, attributesFraction, trainingFraction,
				trainingDataSetAfterPreProcessing, testDataAfterPreProcessing, numberOfTotalOutputs, outputList,
				listOfAllDiscreteValuesBasedOnAttributeNumber);
	}

}
