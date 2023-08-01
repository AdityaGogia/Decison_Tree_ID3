package com.ese589;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class DecisionTreeServiceImpl {
	HashMap<Integer, ArrayList<String>> listOfAllDiscreteValuesBasedOnAttributeNumber;
	List<String> outpultList = new ArrayList<>();
	String output1, output2;
	List<String[]> trainingDataSets;
	List<String[]> testingDataSet;
	DecisionTreeNode decisionTreeNode;
	double trainingTimeForExcecution = 0;
	double precisionValue;
	double recallValue;
	double fscoreValue;
	double accuracyValue;
	int numberOfNodes;

	public DecisionTreeServiceImpl(ArrayList<String[]> trainingDataSetAfterPreProcessing, ArrayList<String[]> testData,
			List<Integer> numberOfTotalOutputs, List<String> outputList,
			HashMap<Integer, ArrayList<String>> listOfAllDiscreteValuesBasedOnAttributeNumber,
			ArrayList<Integer> remainingAttributesToBeAddedInTree) {
		trainingTimeForExcecution = System.currentTimeMillis();

		this.trainingDataSets = trainingDataSetAfterPreProcessing;
		this.output1 = outputList.get(0);
		this.output2 = outputList.get(1);
		this.testingDataSet = testData;
		this.listOfAllDiscreteValuesBasedOnAttributeNumber = listOfAllDiscreteValuesBasedOnAttributeNumber;
		int totalNumberOfOutput1, totalNumberOfOutput2;
		totalNumberOfOutput1 = numberOfTotalOutputs.get(0);
		totalNumberOfOutput2 = numberOfTotalOutputs.get(1);
		decisionTreeNode = new DecisionTreeNode();

		double prob1 = totalNumberOfOutput1 / (totalNumberOfOutput1 + totalNumberOfOutput2 + 0.0);
		double prob2 = totalNumberOfOutput2 / (totalNumberOfOutput1 + totalNumberOfOutput2 + 0.0);
		decisionTreeNode.entropyValue = -1 * calculate_PLogP(prob1) - 1 * calculate_PLogP(prob2);
		decisionTreeNode.dataSet = trainingDataSetAfterPreProcessing;
		decisionTreeNode.numberOfInstancesofOutput1 = totalNumberOfOutput1;
		decisionTreeNode.numberOfInstancesofOutput2 = totalNumberOfOutput2;

		decisionTreeNode.remainingAttributesToBeAddedinDecisionTree = remainingAttributesToBeAddedInTree;
		buildingDecisionTree(decisionTreeNode);

		trainingTimeForExcecution = (System.currentTimeMillis() - trainingTimeForExcecution) / 1000.0;

		decisionTreeAnalysis();
	}

	public void decisionTreeAnalysis() {
		int correctClassificationOutput = 0, incorrectClassificationOutput = 0;
		int tf_TruePositive = 0, fp_falsePositive = 0, fn_falseNegative = 0;
		for (String[] transactionDataSet : testingDataSet) {
			int predictedValue = DecisionTreeNode.predictionOfOutput(decisionTreeNode, transactionDataSet,
					listOfAllDiscreteValuesBasedOnAttributeNumber),
					actualValue = transactionDataSet[transactionDataSet.length - 1].equals(output1) ? 1 : 2;
			if (predictedValue == actualValue) {
				correctClassificationOutput++;
			} else {
				incorrectClassificationOutput++;
			}

			if (predictedValue == 1 && actualValue == 1)
				tf_TruePositive++;
			else if (predictedValue == 1 && actualValue == 2)
				fn_falseNegative++;
			else if (predictedValue == 2 && actualValue == 1)
				fp_falsePositive++;
		}
		precisionValue = tf_TruePositive / (tf_TruePositive + fp_falsePositive + 0.0);
		recallValue = tf_TruePositive / (tf_TruePositive + fn_falseNegative + 0.0);
		fscoreValue = 2 * precisionValue * recallValue / (precisionValue + recallValue);
		accuracyValue = (correctClassificationOutput)
				/ (correctClassificationOutput + incorrectClassificationOutput + 0.0);

		numberOfNodes = 0;
		numberOfNodesOfDecsionTree(decisionTreeNode);
	}

	public void printAnalysisofdecisionTree() {
		System.out.println("No of nodes in tree = " + numberOfNodes);
		System.out.println("Accuracy=" + accuracyValue + "\nPrecision=" + precisionValue + "\nRecall=" + recallValue
				+ "\nF-Score=" + fscoreValue);

	}

	public void printExecutionTimeOfDecisionTree() {
		System.out.println("TrainingTime in secs:" + trainingTimeForExcecution);
	}

	public void numberOfNodesOfDecsionTree(DecisionTreeNode root) {
		if (root == null)
			return;
		numberOfNodes++;
		if (root.leafNode)
			return;
		for (DecisionTreeNode n : root.childNodes)
			numberOfNodesOfDecsionTree(n);
	}

	private static double calculate_PLogP(double x) {
		return x == 0 ? 0 : x * Math.log(x);
	}

	private void buildingDecisionTree(DecisionTreeNode root) {
		if (root == null)
			return;
		if (root.remainingAttributesToBeAddedinDecisionTree.size() == 1) {
			root.leafNode = true;
			root.numberOfClassificationNodes = root.numberOfInstancesofOutput1 >= root.numberOfInstancesofOutput2 ? 1
					: 2;
		} else if (root.numberOfInstancesofOutput1 == 0 || root.numberOfInstancesofOutput2 == 0
				|| root.dataSet.size() == 0) {
			root.leafNode = true;
			root.numberOfClassificationNodes = root.numberOfInstancesofOutput1 == 0 ? 2 : 1;
		} else {
			// find split attribute which gives max gain
			root.childNodes = splitingAttributeUsingTheSelectionMetrics(root);
			ArrayList<String> discreteValuesOfThisAttribute = listOfAllDiscreteValuesBasedOnAttributeNumber
					.get(root.currentAttributeNumber_ColumnNumber);
			for (int j = 0; j < discreteValuesOfThisAttribute.size(); j++) {
				root.childNodes[j].dataSet = new ArrayList<>();
				root.childNodes[j].remainingAttributesToBeAddedinDecisionTree = new ArrayList<>();
				for (int rem : root.remainingAttributesToBeAddedinDecisionTree) {
					if (rem != root.currentAttributeNumber_ColumnNumber)
						root.childNodes[j].remainingAttributesToBeAddedinDecisionTree.add(rem);
				}
				String curr = discreteValuesOfThisAttribute.get(j);
				for (String[] s : root.dataSet) {
					if (s[root.currentAttributeNumber_ColumnNumber].equals(curr)) {
						root.childNodes[j].dataSet.add(s);
					}
				}
				buildingDecisionTree(root.childNodes[j]);
			}
		}
	}

	public DecisionTreeNode[] splitingAttributeUsingTheSelectionMetrics(DecisionTreeNode decisionTreeNode) {
		double maximumGain = -1.0;
		double maximumGini = -1.0;
		DecisionTreeNode[] nodeToBeSeleceted = null;
		double gini_Index = 0 ;
		for (int i : decisionTreeNode.remainingAttributesToBeAddedinDecisionTree) {
			ArrayList<String> listOfDiscreteValuesOfThisAttribute = listOfAllDiscreteValuesBasedOnAttributeNumber.get(i);
			DecisionTreeNode[] childNode = new DecisionTreeNode[listOfDiscreteValuesOfThisAttribute.size()];
			for (int j = 0; j < listOfDiscreteValuesOfThisAttribute.size(); j++) {
				String present = listOfDiscreteValuesOfThisAttribute.get(j);
				childNode[j] = new DecisionTreeNode();
				for (String[] tem : decisionTreeNode.dataSet) {
					if (tem[i].equals(present)) {
						if (tem[tem.length - 1].equals(output1))
							childNode[j].numberOfInstancesofOutput1++;
						else
							childNode[j].numberOfInstancesofOutput2++;
					}
				}
			}
			int totalValue = decisionTreeNode.dataSet.size();
			double gain = decisionTreeNode.entropyValue;
			for (int j = 0; j < listOfDiscreteValuesOfThisAttribute.size(); j++) {
				int outputClass1 = childNode[j].numberOfInstancesofOutput1, outputClass2 = childNode[j].numberOfInstancesofOutput2;
				if (outputClass1 == 0 && outputClass2 == 0)
					continue;
				double prob1 = outputClass1 / (outputClass1 + outputClass2 + 0.0), prob2 = outputClass2 / (outputClass1 + outputClass2 + 0.0);
				childNode[j].entropyValue = -1 * calculate_PLogP(prob1) + -1 * calculate_PLogP(prob2);
				gini_Index = 1 - Math.pow(prob1, 2) -  Math.pow(prob2, 2);
			//	gain -= ((outputClass1 + outputClass2) / (totalValue + 0.0)) * childNode[j].entropyValue;
			}
//			if (gain > maximumGain) {
//				decisionTreeNode.currentAttributeNumber_ColumnNumber = i;
//				maximumGain = gain;
//				nodeToBeSeleceted = childNode;
//			}
			if(gini_Index>maximumGini) {
				decisionTreeNode.currentAttributeNumber_ColumnNumber = i;
				maximumGini = gini_Index;
				nodeToBeSeleceted = childNode;
			}
		}
		return nodeToBeSeleceted;
	}
}
