console.log("limdu batch learning start");

//This example uses the Naive Bayes implementation [classifier.js, by Heather Arthur](https://github.com/harthur/classifier).

function newClassifierFunction() {
	//var limdu = require(__dirname + '/../limdu');
	var limdu = require('limdu');
	return new limdu.classifiers.Bayesian({weight:0.001});
}

var colorClassifier = newClassifierFunction();

colorClassifier.trainBatch([
	{input: { r: 8, g: 2, b: 7 }, output: 'purple'},
//	{input: { r: 0.16, g: 0.09, b: 0.2 }, output: 'white'},
	{input: { r: 6, g: 7, b: 1 }, output: 'yellow'},
	]);

console.log("\nBefore serialization:")
console.log("{ r:9, g:3, b:8 } = ",
	colorClassifier.classify({ r:9, g:3, b:8 },
		/* explanation level = */4));
console.log("{ r:7, g:8, b:2 } = ",
  colorClassifier.classify({ r:7, g:8, b:2 },
		/* explanation level = */4));

// Serialization demo:
var serialize = require('serialization');
var colorClassifierString = serialize.toString(colorClassifier, newClassifierFunction);

console.log("\nSerialization:")
console.log(colorClassifierString);

var colorClassifierCopy = serialize.fromString(colorClassifierString, __dirname);

console.log("\nAfter serialization:")
console.log("{ r:9, g:3, b:8 } = ",
	colorClassifierCopy.classify({ r:9, g:3, b:8 },
		/* explanation level = */1));
console.log("{ r:7, g:8, b:2 } = ",
  colorClassifierCopy.classify({ r:7, g:8, b:2 },
		/* explanation level = */1));


console.log("limdu batch learning end");
