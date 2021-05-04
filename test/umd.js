const neuro = require('../dist/index.umd.js');

// First, define our base classifier type (a multi-label classifier based on winnow):
var TextClassifier = neuro.classifiers.multilabel.BinaryRelevance.bind(0, {
	binaryClassifierType: neuro.classifiers.Winnow.bind(0, { retrain_count: 10 })
});

// Now define our feature extractor - a function that takes a sample and adds features to a given features set:
var WordExtractor = function(input, features) {
	input.split(" ").forEach(function(word) {
		features[word] = 1;
	});
};

// Initialize a classifier with the base classifier type and the feature extractor:
var intentClassifier = new neuro.classifiers.EnhancedClassifier({
	classifierType: TextClassifier,
	featureExtractor: WordExtractor
});

// Train and test:
intentClassifier.trainBatch([
	{ input: "안녕하세요.", output: "안녕하세요." },
	{ input: "반가워요", output: "방가" },
]);

console.dir(intentClassifier.classify("안녕"));
console.dir(intentClassifier.classify("안녕하세요."));
console.dir(intentClassifier.classify("방가방가"));
console.dir(intentClassifier.classify("방가"));
console.dir(intentClassifier.classify("반가워요"));

