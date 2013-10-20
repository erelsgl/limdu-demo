console.log("limdu back classification demo start");

var limdu = require('limdu');

// Initialize a multi-label classifier with a feature extractor and past-training-samples:
var intentClassifier = new limdu.classifiers.EnhancedClassifier({
	classifierType: limdu.classifiers.multilabel.BinaryRelevance.bind(0, {
		binaryClassifierType: limdu.classifiers.Winnow.bind(0, {retrain_count: 10})
	}),
	featureExtractor: limdu.features.NGramsOfWords(1),
	pastTrainingSamples: [],
});

// Train and test:
intentClassifier.trainBatch([
	{input: "I want an apple", output: "apl"},
	{input: "I want a banana", output: "bnn"},
	{input: "I really want an apple", output: "apl"},
	{input: "I want a banana very much", output: "bnn"},
	]);

console.dir(intentClassifier.backClassify("apl"));  // [ 'I want an apple', 'I really want an apple' ]

console.log("limdu back classification demo end");
