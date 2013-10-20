console.log("limdu lookup table demo start");

//This example uses the quadratic SVM implementation [svm.js, by Andrej Karpathy](https://github.com/karpathy/svmjs).

var limdu = require('limdu');

// First, define our base classifier type (a multi-label classifier based on svm.js):
var TextClassifier = limdu.classifiers.multilabel.BinaryRelevance.bind(0, {
	binaryClassifierType: limdu.classifiers.SvmJs.bind(0, {C: 1.0})
});

// Initialize a classifier with a feature extractor and a lookup table:
var intentClassifier = new limdu.classifiers.EnhancedClassifier({
	classifierType: TextClassifier,
	featureExtractor: limdu.features.NGramsFromText(1),  // each word ("1-gram") is a feature  
	featureLookupTable: new limdu.features.FeatureLookupTable()
});

// Train and test:
intentClassifier.trainBatch([
	{input: "I want an apple", output: "apl"},
	{input: "I want a banana", output: "bnn"},
	{input: "I want chips", output: "cps"},
	]);

console.dir(intentClassifier.classify("I want an apple and a banana"));  // ['apl','bnn']

console.log("limdu lookup table demo end");
