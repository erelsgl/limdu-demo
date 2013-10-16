console.log("limdu demo start");

var limdu = require('limdu');

// Use binding to define our classifier type (a multi-label classifier based on winnow):
var TextClassifier = limdu.classifiers.multilabel.BinaryRelevance.bind(0, {
	binaryClassifierType: limdu.classifiers.Winnow.bind(0, {retrain_count: 10})
});

// Define a feature extractor (a function that takes a sample and add features to a given features set):
var WordExtractor = function(input, features) {
	input.split(" ").forEach(function(word) {
		features[word]=1;
	});
};

// Initialize a classifier with a feature extractor:
var intentClassifier = new limdu.classifiers.EnhancedClassifier({
	classifierType: TextClassifier,
	featureExtractor: WordExtractor
});

// Train and test:
intentClassifier.trainBatch([
	{input: "I want an apple", output: "apl"},
	{input: "I want a banana", output: "bnn"},
	{input: "I want chips", output: "CHIPS"},
	]);

console.dir(intentClassifier.classify("I want an apple and a banana"));  // ['apl','bnn']
console.dir(intentClassifier.classify("I WANT AN APPLE AND A BANANA"));  // [] (case sensitive)

//Initialize a classifier with a feature extractor and a case normalizer:
intentClassifier = new limdu.classifiers.EnhancedClassifier({
	classifierType: TextClassifier,
	normalizer: limdu.features.LowerCaseNormalizer,
	featureExtractor: WordExtractor
});

//Train and test:
intentClassifier.trainBatch([
	{input: "I want an apple", output: "apl"},
	{input: "I want a banana", output: "bnn"},
	{input: "I want chips", output: "CHIPS"},
	]);

console.dir(intentClassifier.classify("I want an apple and a banana"));  // ['apl','bnn']
console.dir(intentClassifier.classify("I WANT AN APPLE AND A BANANA"));  // ['apl','bnn'] (case insensitive)

console.log("limdu demo end");
