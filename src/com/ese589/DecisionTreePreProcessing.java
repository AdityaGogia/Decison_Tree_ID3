package com.ese589;

import java.util.Collections;
import java.io.FileReader;
import java.util.HashMap;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.StringTokenizer;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.BufferedReader;

public class DecisionTreePreProcessing {
	List<Integer> numberOfOutputs = new ArrayList<>();
	int numberOfOutput1, numberOfOutput2;
	static int[] continousValuesToDiscreteValues;
	String output1, output2;
	List<Integer> columnNumberForContinuousAttributesToMakeItDiscrete;

	static HashMap<Integer, ArrayList<String>> listOfAllDiscreteValuesBasedOnAttributeNumber = new HashMap<Integer, ArrayList<String>>();

	public DecisionTreePreProcessing(List<String> outputList,
			List<Integer> columnNumberForContinuousAttributesToMakeItDiscrete) {
		this.output1 = outputList.get(0);
		this.output2 = outputList.get(0);
		this.columnNumberForContinuousAttributesToMakeItDiscrete = columnNumberForContinuousAttributesToMakeItDiscrete;
	}

	public ArrayList<String[]> preProcessingDataSetToMakeItDiscrete(File trainingDataSet) throws IOException {
		BufferedReader bufferedReader = new BufferedReader(new FileReader(trainingDataSet));

		String transactionData = bufferedReader.readLine();
		bufferedReader.close();
		StringTokenizer stringTokenizer = new StringTokenizer(transactionData, ",");
		int numberOfAttributesInATransaction = stringTokenizer.countTokens();
		continousValuesToDiscreteValues = new int[numberOfAttributesInATransaction];
		ArrayList<String[]> dataSetAfterDiscretise = new ArrayList<>();
		bufferedReader = new BufferedReader(new FileReader(trainingDataSet));
		while ((transactionData = bufferedReader.readLine()) != null) {
			stringTokenizer = new StringTokenizer(transactionData, ",");
			String[] transactionDataSet = new String[numberOfAttributesInATransaction];
			for (int i = 0; i < numberOfAttributesInATransaction; i++) {
				transactionDataSet[i] = stringTokenizer.nextToken();
			}
			if (transactionDataSet[numberOfAttributesInATransaction - 1].equals(output1)) {
				numberOfOutput1++;
			} else {
				numberOfOutput2++;
			}
			dataSetAfterDiscretise.add(transactionDataSet);
		}
		numberOfOutputs.add(numberOfOutput1);
		numberOfOutputs.add(numberOfOutput2);
		for (int i : columnNumberForContinuousAttributesToMakeItDiscrete) {
			int numberOfOutputtoLeftofOutput1 = 0, numberOfOutputtoRightofOutput1 = numberOfOutput1;
			int numberOfOutputtoLeftofOutput2 = 0, numberOfOutputtoRightofOutput2 = numberOfOutput2;
			int totalNumberOfDataSet = numberOfOutput1 + numberOfOutput2;
			// We Actually Sort the data set based on the value of the continuous data.
			Collections.sort(dataSetAfterDiscretise, new DecisionTreeComparator(i));
			double minimumEntropyValue = Double.MAX_VALUE;
			int partionValue = 1;
			String back = dataSetAfterDiscretise.get(0)[i];
			for (String[] transactionDataset : dataSetAfterDiscretise) {
				if (transactionDataset[numberOfAttributesInATransaction - 1].equals(output1)) {
					numberOfOutputtoLeftofOutput1++;
					numberOfOutputtoRightofOutput1--;
				} else {
					numberOfOutputtoLeftofOutput2++;
					numberOfOutputtoRightofOutput2--;
				}
				String present = transactionDataset[i];
				if (present.equals(back) || present.equals("?"))
					continue;
				double prob11 = (numberOfOutputtoLeftofOutput1 + 0.0)
						/ (numberOfOutputtoLeftofOutput1 + numberOfOutputtoLeftofOutput2),
						prob12 = (numberOfOutputtoLeftofOutput2 + 0.0)
								/ (numberOfOutputtoLeftofOutput1 + numberOfOutputtoLeftofOutput2);
				double prob21 = (numberOfOutputtoRightofOutput1 + 0.0)
						/ (numberOfOutputtoRightofOutput1 + numberOfOutputtoRightofOutput2),
						prob22 = (numberOfOutputtoRightofOutput2 + 0.0)
								/ (numberOfOutputtoRightofOutput1 + numberOfOutputtoRightofOutput2);
				double currentEntropyValue = ((numberOfOutputtoLeftofOutput1 + numberOfOutputtoLeftofOutput2)
						/ totalNumberOfDataSet) * (-1 * prob11 * Math.log(prob11) - 1 * prob12 * Math.log(prob12))
						+ ((numberOfOutputtoRightofOutput1 + numberOfOutputtoRightofOutput2) / totalNumberOfDataSet)
								* (-1 * prob21 * Math.log(prob21) - 1 * prob22 * Math.log(prob22));
				if (currentEntropyValue < minimumEntropyValue) {
					minimumEntropyValue = currentEntropyValue;
					partionValue = (Integer.parseInt(back) + Integer.parseInt(present)) / 2;
					continousValuesToDiscreteValues[i] = partionValue;
				}
				back = present;
			}
			String updateColumn1Name = "<=" + partionValue, updateColumn2Name = ">" + partionValue;
			for (String[] dataset : dataSetAfterDiscretise) {
				if (dataset[i] == "?")
					continue;
				if (Integer.parseInt(dataset[i]) <= partionValue)
					dataset[i] = updateColumn1Name;
				else
					dataset[i] = updateColumn2Name;
			}
		}
		bufferedReader.close();
		return dataSetAfterDiscretise;
	}

	public static void predictMissingOrEmptyValues(ArrayList<String[]> dataSetAfterDiscretise, String temp)
			throws IOException {
		int n = dataSetAfterDiscretise.get(0).length - 1;
		String[] predictedValue = new String[n];
		for (int i = 0; i < n; i++) {
			HashMap<String, Integer> count = new HashMap<>();
			for (String[] s : dataSetAfterDiscretise) {
				if (s[i].equals("?"))
					continue;
				if (count.containsKey(s[i]))
					count.put(s[i], count.get(s[i]) + 1);
				else
					count.put(s[i], 1);
			}
			int max = 0;
			for (String s : count.keySet()) {
				int c = count.get(s);
				if (c > max) {
					max = c;
					predictedValue[i] = s;
				}
			}
		}

		for (String[] s : dataSetAfterDiscretise) {
			for (int i = 0; i < n; i++) {
				if (s[i].equals("?"))
					s[i] = predictedValue[i];
			}
		}

		File file = new File(temp + "_PreProcessed.data");
		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file));
		for (String[] transactionDataset : dataSetAfterDiscretise) {
			bufferedWriter.write(
					Arrays.toString(transactionDataset).replace("[", "").replace("]", "").replace(" ", "") + "\n");
		}
		bufferedWriter.close();
	}

	public static ArrayList<String[]> preProcessingDataSetToMakeItDiscreteTestData(File testData,
			List<Integer> columnNumberForContinuousAttributesToMakeItDiscrete) throws IOException {
		BufferedReader bufferedReader = new BufferedReader(new FileReader(testData));

		String transactionData = bufferedReader.readLine();
		bufferedReader.close();
		StringTokenizer stringTokenizer = new StringTokenizer(transactionData, ",");
		int numberOfColumns = stringTokenizer.countTokens();
		ArrayList<String[]> finalTestData = new ArrayList<>();
		bufferedReader = new BufferedReader(new FileReader(testData));
		while ((transactionData = bufferedReader.readLine()) != null) {
			stringTokenizer = new StringTokenizer(transactionData, ",");
			String[] transactionDataSet = new String[numberOfColumns];
			for (int i = 0; i < numberOfColumns; i++) {
				transactionDataSet[i] = stringTokenizer.nextToken();
			}
			finalTestData.add(transactionDataSet);
		}
		for (int i : columnNumberForContinuousAttributesToMakeItDiscrete) {
			String newc1Name = "<=" + continousValuesToDiscreteValues[i],
					newc2Name = ">" + continousValuesToDiscreteValues[i];
			for (String[] dataset : finalTestData) {
				if (dataset[i] == "?")
					continue;
				if (Integer.parseInt(dataset[i]) <= continousValuesToDiscreteValues[i])
					dataset[i] = newc1Name;
				else
					dataset[i] = newc2Name;
			}
		}
		bufferedReader.close();
		predictMissingOrEmptyValues(finalTestData, "test");
		return finalTestData;
	}

	public static void computelistOfAllDiscreteValuesForEachColumn(ArrayList<String[]> finalDataSet) {
		HashSet<String> differentAttributesList = new HashSet<>();
		String[] firstTransaction = finalDataSet.get(0);
		for (int i = 0; i < firstTransaction.length - 1; i++) {
			listOfAllDiscreteValuesBasedOnAttributeNumber.put(i, new ArrayList<String>());
		}
		for (String[] transactionDataset : finalDataSet) {
			for (int i = 0; i < transactionDataset.length - 1; i++) {
				String value = transactionDataset[i];
				if (value.equals("?"))
					continue;
				if (!differentAttributesList.contains(value)) {
					listOfAllDiscreteValuesBasedOnAttributeNumber.get(i).add(value);
					differentAttributesList.add(value);
				}
			}
		}
	}
}

class DecisionTreeComparator implements Comparator<String[]> {
	int i;

	public DecisionTreeComparator(int i) {
		this.i = i;
	}

	public int compare(String[] string1, String[] string2) {
		return string1[i].compareTo(string2[i]);
	}
}