"use strict";

var _SvmJs = _interopRequireDefault(require("./core/svm/SvmJs"));

var _WinnowHash = _interopRequireDefault(require("./core/winnow/WinnowHash"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }

var _require = require("./core/EnhancedClassifier"),
    EnhancedClassifier = _require.EnhancedClassifier;

var multilabel = require("./core/multilabel");

var NeuralNetwork = require("./core/neural/NeuralNetwork");

var SvmLinear = require("./core/svm/SvmLinear");

var SvmPerf = require("./core/svm/SvmPerf");

var features = require("./features");

var formats = require("./formats");

var utils = require("./utils");

module.exports = {
  classifiers: {
    NeuralNetwork: NeuralNetwork,
    SvmJs: _SvmJs["default"],
    SvmLinear: SvmLinear,
    SvmPerf: SvmPerf,
    Winnow: _WinnowHash["default"],
    multilabel: multilabel,
    EnhancedClassifier: EnhancedClassifier
  },
  features: features,
  formats: formats,
  utils: utils
};