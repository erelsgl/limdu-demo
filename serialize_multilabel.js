console.log("limdu serialize demo start");

var serialize = require('serialization');

// First, define a function that creates a fresh  (untrained) classifier.
// This code should be stand-alone - it should include all the 'require' statements
//   required for creating the classifier.
function newClassifierFunction() {
	var limdu = require('limdu');
	var TextClassifier = limdu.classifiers.multilabel.BinaryRelevance.bind(0, {
		binaryClassifierType: limdu.classifiers.Winnow.bind(0, {retrain_count: 10})
	});

	var WordExtractor = function(input, features) {
		input.split(" ").forEach(function(word) {
			features[word]=1;
		});
	};
	
	// Initialize a classifier with a feature extractor:
	return new limdu.classifiers.EnhancedClassifier({
		classifierType: TextClassifier,
		featureExtractor: WordExtractor,
		pastTrainingSamples: [], // to enable retraining
	});
}

// Use the above function for creating a new classifier:
var intentClassifier = newClassifierFunction();

// Train and test:
intentClassifier.trainBatch([
	{input: "I want an apple", output: "apl"},
	{input: "I want a banana", output: "bnn"},
	{input: "I want chips", output: "cps"},
	]);

console.log("Original classifier:");
intentClassifier.classifyAndLog("I want an apple and a banana");  // ['apl','bnn']
intentClassifier.trainOnline("I want a doughnut", "dnt");
intentClassifier.classifyAndLog("I want chips and a doughnut");  // ['cps','dnt']
intentClassifier.retrain();
intentClassifier.classifyAndLog("I want an apple and a banana");  // ['apl','bnn']
intentClassifier.classifyAndLog("I want chips and a doughnut");  // ['cps','dnt']

// Serialize the classifier (convert it to a string)
var intentClassifierString = serialize.toString(intentClassifier, newClassifierFunction);

// Save the string to a file, and send it to a remote server.

// ...

// On the remote server, retrieve the string from a file and then:

var intentClassifierCopy = serialize.fromString(intentClassifierString, __dirname);

console.log("Deserialized classifier:");
intentClassifierCopy.classifyAndLog("I want an apple and a banana");  // ['apl','bnn']
intentClassifierCopy.classifyAndLog("I want chips and a doughnut");  // ['cps','dnt']
intentClassifierCopy.trainOnline("I want an elm tree", "elm");
intentClassifierCopy.classifyAndLog("I want doughnut and elm tree");  // ['dnt','elm']

console.log("limdu feature engineering end");
