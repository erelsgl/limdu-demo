console.log("limdu batch learning start");

//This example uses the Naive Bayes implementation [classifier.js, by Heather Arthur](https://github.com/harthur/classifier).

var limdu = require('limdu');

var colorClassifier = new limdu.classifiers.Bayesian();

colorClassifier.trainBatch([
	{input: { r: 0.03, g: 0.7, b: 0.5 }, output: 'black'}, 
	{input: { r: 0.16, g: 0.09, b: 0.2 }, output: 'white'},
	{input: { r: 0.5, g: 0.5, b: 1.0 }, output: 'white'},
	]);

console.log(colorClassifier.classify({ r: 1, g: 0.4, b: 0 }, 1));  // [0.987]

console.log("limdu batch learning end");
