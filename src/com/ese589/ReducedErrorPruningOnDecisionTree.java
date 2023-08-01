package com.ese589;

public class ReducedErrorPruningOnDecisionTree {
	DecisionTreeServiceImpl decisionTree;
	double  maximumAccuracy;
	double accuracyOfDecsionTreeBeforeREP;

	public ReducedErrorPruningOnDecisionTree(DecisionTreeServiceImpl decisionTree) {
		this.decisionTree = decisionTree;
		accuracyOfDecsionTreeBeforeREP = decisionTree.accuracyValue;
		maximumAccuracy = accuracyOfDecsionTreeBeforeREP;
		double trainingTimeStart = System.currentTimeMillis();
		pruneDecsionTreeNode(decisionTree.decisionTreeNode);
		double trainingTimeEnd = System.currentTimeMillis();
		double totalTimeTaken = (trainingTimeEnd - trainingTimeStart) / 1000.0;
		System.out.println("Training Time in seconds=" + totalTimeTaken);
	}

	private void pruneDecsionTreeNode(DecisionTreeNode decisionTreeNode) {
		if (decisionTreeNode == null)
			return;
		decisionTreeNode.leafNode = true;
		decisionTreeNode.numberOfClassificationNodes = decisionTreeNode.numberOfInstancesofOutput1 >= decisionTreeNode.numberOfInstancesofOutput2 ? 1 : 2;
		decisionTree.decisionTreeAnalysis();
		if (decisionTree.accuracyValue > maximumAccuracy) {
			maximumAccuracy = decisionTree.accuracyValue;
			return;
		}
		decisionTreeNode.leafNode = false;
		for (DecisionTreeNode treeNode : decisionTreeNode.childNodes) {
			if (treeNode.leafNode)
				continue;
			pruneDecsionTreeNode(treeNode);
		}

	}
}
