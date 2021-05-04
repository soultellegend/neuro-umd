const { EnhancedClassifier } = require("./core/EnhancedClassifier");
const multilabel = require("./core/multilabel");
const NeuralNetwork = require("./core/neural/NeuralNetwork");
import SvmJs from "./core/svm/SvmJs";
const SvmLinear = require("./core/svm/SvmLinear");
const SvmPerf = require("./core/svm/SvmPerf");
import Winnow from "./core/winnow/WinnowHash";
const features = require("./features");
const formats = require("./formats");
const utils = require("./utils");

module.exports = {
  classifiers: {
    NeuralNetwork,
    SvmJs,
    SvmLinear,
    SvmPerf,
    Winnow,
    multilabel,
    EnhancedClassifier
  },
  features,
  formats,
  utils
};
