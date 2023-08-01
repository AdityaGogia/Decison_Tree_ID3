package com.ese589;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class DecisionTreeNode {

	int currentAttributeNumber_ColumnNumber;
	List<Integer> remainingAttributesToBeAddedinDecisionTree;
	int numberOfInstancesofOutput1;
	int numberOfInstancesofOutput2;
	double entropyValue;
	DecisionTreeNode[] childNodes;
	List<String[]> dataSet;
	boolean leafNode = false;
	int numberOfClassificationNodes;

	public DecisionTreeNode(int numberOfClassificationNodes) {
		leafNode = true;
		this.numberOfClassificationNodes = numberOfClassificationNodes;
	}

	public DecisionTreeNode() {

	}

	public static int predictionOfOutput(DecisionTreeNode decisionTreeNode, String[] transactionData,
			HashMap<Integer, ArrayList<String>> listOfAllDiscreteValuesBasedOnAttributeNumber) {
		if (decisionTreeNode == null) {
			return 1;
		}
		if (decisionTreeNode.leafNode) {
			return decisionTreeNode.numberOfClassificationNodes;
		}
		String temp = transactionData[decisionTreeNode.currentAttributeNumber_ColumnNumber];
		ArrayList<String> discreteValues = listOfAllDiscreteValuesBasedOnAttributeNumber
				.get(decisionTreeNode.currentAttributeNumber_ColumnNumber);
		for (int i = 0; i < discreteValues.size(); i++) {
			if (temp.equals(discreteValues.get(i))) {
				return predictionOfOutput(decisionTreeNode.childNodes[i], transactionData,
						listOfAllDiscreteValuesBasedOnAttributeNumber);
			}
		}
		return 1;
	}
}
