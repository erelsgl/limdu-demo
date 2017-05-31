console.log("limdu cross-validation demo start");

var limdu = require("../limdu");

// define a toy dataset:
var dataset = [
           	{input: "I want an apple", output: "apl"},
        	{input: "I want a banana", output: "bnn"},
        	{input: "I want chips", output: "cps"},
           	{input: "I want an apple and a banana", output: ["apl","bnn"]},
        	{input: "I want a banana and chips", output: ["bnn","cps"]},
        	{input: "I want chips and an apple", output: ["cps","apl"]},
           	{input: "I want nothing", output: []},
        	{input: "I want a banana and chips and an apple", output: ["apl","bnn","cps"]},
        	{input: "I want chips and a banana and and an apple", output: ["apl","bnn","cps"]},
        	{input: "I want an apple and chips and a banana", output: ["apl","bnn","cps"]},
        	];
var numOfFolds = 5; // for k-fold cross-validation

// Define the type of classifier that we want to test:
var IntentClassifier = limdu.classifiers.EnhancedClassifier.bind(0, {
	classifierType: limdu.classifiers.multilabel.BinaryRelevance.bind(0, {
		binaryClassifierType: limdu.classifiers.Winnow.bind(0, {retrain_count: 10})
	}),
	featureExtractor: limdu.features.NGramsOfWords(1),
});

var microAverage = new limdu.utils.PrecisionRecall();
var macroAverage = new limdu.utils.PrecisionRecall();

var verbosity = 0;

limdu.utils.partitions.partitions(dataset, numOfFolds, function(trainSet, testSet) {
	console.log("Training on "+trainSet.length+" samples, testing on "+testSet.length+" samples");
	var classifier = new IntentClassifier();
	classifier.trainBatch(trainSet);
	limdu.utils.test(classifier, testSet, verbosity,
		microAverage, macroAverage);
});

macroAverage.calculateMacroAverageStats(numOfFolds);
console.log("\n\nMACRO AVERAGE:"); console.dir(macroAverage.fullStats());

microAverage.calculateStats();
console.log("\n\nMICRO AVERAGE:"); console.dir(microAverage.fullStats());

console.log("limdu cross-validation demo end");
