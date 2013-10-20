console.log("limdu multi label demo start");

// This example demonstrates the binary-relevance (aka one-vs-all) multi-label classification method.
// The base binary classifier is Winnow, with a pre-specified parameter set. 

var limdu = require('limdu');

var MyWinnow = limdu.classifiers.Winnow.bind(0, {
	retrain_count: 10,
});

var intentClassifier = new limdu.classifiers.multilabel.BinaryRelevance({
	binaryClassifierType: MyWinnow,
});

intentClassifier.trainBatch([
	{input: {I:1,want:1,an:1,apple:1}, output: "APPLE"},
	{input: {I:1,want:1,a:1,banana:1}, output: "BANANA"},
	{input: {I:1,want:1,chips:1}, output: "CHIPS"},
	]);

console.dir(intentClassifier.classify({I:1,want:1,an:1,apple:1,and:1,a:1,banana:1}));  // ['APPLE','BANANA']

console.log("limdu multi label demo end");
