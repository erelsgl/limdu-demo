console.log("limdu demo start");

var createNewClassifier = function() {
	var limdu = require('limdu');
	return new limdu.classifiers.Winnow({
		default_positive_weight: 1,
		default_negative_weight: 1,
		threshold: 0,
	});
};

var limdu = require('limdu');

var birdClassifier = createNewClassifier();
birdClassifier.trainOnline({'wings': 1, 'flight': 1, 'beak': 1, 'eagle': 1}, 1);  
birdClassifier.trainOnline({'wings': 0, 'flight': 0, 'beak': 0, 'dog': 1}, 0);    
birdClassifier.trainOnline({'wings': 1, 'flight': 0, 'beak': 1, 'penguin':1}, 1);   
birdClassifier.trainOnline({'wings': 0, 'flight': 1, 'beak': 0, 'bat': 1}, 0);

console.log("\nORIGINAL TRAINED CLASSIFIER: ");
console.dir(birdClassifier.classify({'wings': 1, 'flight': 0, 'beak': 1, 'chicken': 1})); // 1  
console.dir(birdClassifier.classify({'wings': 0, 'flight': 0, 'beak': 0, 'cat': 1})); // 0

// Convert the classifier to a string (you can save this string in a file, if you want):
var birdClassifierString = limdu.utils.toString(birdClassifier, createNewClassifier);

// Create a new classifier from that string:
var birdClassifier2 = limdu.utils.fromString(birdClassifierString, __dirname);

console.log("\nDESERIALIZED CLASSIFIER: ");
console.dir(birdClassifier2.classify({'wings': 1, 'flight': 0, 'beak': 1, 'chicken': 1})); // 1  
console.dir(birdClassifier2.classify({'wings': 0, 'flight': 0, 'beak': 0, 'cat': 1})); // 0

console.log("limdu demo end");
