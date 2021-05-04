(function (factory) {
  typeof define === 'function' && define.amd ? define(factory) :
  factory();
}((function () { 'use strict';

  function _classCallCheck(instance, Constructor) {
    if (!(instance instanceof Constructor)) {
      throw new TypeError("Cannot call a class as a function");
    }
  }

  function createCommonjsModule(fn, module) {
  	return module = { exports: {} }, fn(module, module.exports), module.exports;
  }

  var svm = createCommonjsModule(function (module) {
  // MIT License
  // Andrej Karpathy

  ((function(exports){

    /*
      This is a binary SVM and is trained using the SMO algorithm.
      Reference: "The Simplified SMO Algorithm" (http://math.unt.edu/~hsp0009/smo.pdf)
      
      Simple usage example:
      svm = svmjs.SVM();
      svm.train(data, labels);
      testlabels = svm.predict(testdata);
    */
    var SVM = function(options) {
    };

    SVM.prototype = {
      
      // data is NxD array of floats. labels are 1 or -1.
      train: function(data, labels, options) {
        
        // we need these in helper functions
        this.data = data;
        this.labels = labels;

        // parameters
        options = options || {};
        var C = options.C || 1.0; // C value. Decrease for more regularization
        var tol = options.tol || 1e-4; // numerical tolerance. Don't touch unless you're pro
        var alphatol = options.alphatol || 1e-7; // non-support vectors for space and time efficiency are truncated. To guarantee correct result set this to 0 to do no truncating. If you want to increase efficiency, experiment with setting this little higher, up to maybe 1e-4 or so.
        var maxiter = options.maxiter || 10000; // max number of iterations
        var numpasses = options.numpasses || 10; // how many passes over data with no change before we halt? Increase for more precision.
        
        // instantiate kernel according to options. kernel can be given as string or as a custom function
        var kernel = linearKernel;
        this.kernelType = "linear";
        if("kernel" in options) {
          if(typeof options.kernel === "string") {
            // kernel was specified as a string. Handle these special cases appropriately
            if(options.kernel === "linear") { 
              this.kernelType = "linear"; 
              kernel = linearKernel; 
            }
            if(options.kernel === "rbf") { 
              var rbfSigma = options.rbfsigma || 0.5;
              this.rbfSigma = rbfSigma; // back this up
              this.kernelType = "rbf";
              kernel = makeRbfKernel(rbfSigma);
            }
          } else {
            // assume kernel was specified as a function. Let's just use it
            this.kernelType = "custom";
            kernel = options.kernel;
          }
        }

        // initializations
        this.kernel = kernel;
        this.N = data.length; var N = this.N;
        this.D = data[0].length; this.D;
        this.alpha = zeros(N);
        this.b = 0.0;
        this.usew_ = false; // internal efficiency flag

        // run SMO algorithm
        var iter = 0;
        var passes = 0;
        while(passes < numpasses && iter < maxiter) {
          
          var alphaChanged = 0;
          for(var i=0;i<N;i++) {
          
            var Ei= this.marginOne(data[i]) - labels[i];
            if( (labels[i]*Ei < -tol && this.alpha[i] < C)
             || (labels[i]*Ei > tol && this.alpha[i] > 0) ){
              
              // alpha_i needs updating! Pick a j to update it with
              var j = i;
              while(j === i) j= randi(0, this.N);
              var Ej= this.marginOne(data[j]) - labels[j];
              
              // calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
              ai= this.alpha[i];
              aj= this.alpha[j];
              var L = 0; var H = C;
              if(labels[i] === labels[j]) {
                L = Math.max(0, ai+aj-C);
                H = Math.min(C, ai+aj);
              } else {
                L = Math.max(0, aj-ai);
                H = Math.min(C, C+aj-ai);
              }
              
              if(Math.abs(L - H) < 1e-4) continue;
              
              var eta = 2*kernel(data[i],data[j]) - kernel(data[i],data[i]) - kernel(data[j],data[j]);
              if(eta >= 0) continue;
              
              // compute new alpha_j and clip it inside [0 C]x[0 C] box
              // then compute alpha_i based on it.
              var newaj = aj - labels[j]*(Ei-Ej) / eta;
              if(newaj>H) newaj = H;
              if(newaj<L) newaj = L;
              if(Math.abs(aj - newaj) < 1e-4) continue; 
              this.alpha[j] = newaj;
              var newai = ai + labels[i]*labels[j]*(aj - newaj);
              this.alpha[i] = newai;
              
              // update the bias term
              var b1 = this.b - Ei - labels[i]*(newai-ai)*kernel(data[i],data[i])
                       - labels[j]*(newaj-aj)*kernel(data[i],data[j]);
              var b2 = this.b - Ej - labels[i]*(newai-ai)*kernel(data[i],data[j])
                       - labels[j]*(newaj-aj)*kernel(data[j],data[j]);
              this.b = 0.5*(b1+b2);
              if(newai > 0 && newai < C) this.b= b1;
              if(newaj > 0 && newaj < C) this.b= b2;
              
              alphaChanged++;
              
            } // end alpha_i needed updating
          } // end for i=1..N
          
          iter++;
          //console.log("iter number %d, alphaChanged = %d", iter, alphaChanged);
          if(alphaChanged == 0) passes++;
          else passes= 0;
          
        } // end outer loop
        
        // if the user was using a linear kernel, lets also compute and store the
        // weights. This will speed up evaluations during testing time
        if(this.kernelType === "linear") {

          // compute weights and store them
          this.w = new Array(this.D);
          for(var j=0;j<this.D;j++) {
            var s= 0.0;
            for(var i=0;i<this.N;i++) {
              s+= this.alpha[i] * labels[i] * data[i][j];
            }
            this.w[j] = s;
            this.usew_ = true;
          }
        } else {

          // okay, we need to retain all the support vectors in the training data,
          // we can't just get away with computing the weights and throwing it out

          // But! We only need to store the support vectors for evaluation of testing
          // instances. So filter here based on this.alpha[i]. The training data
          // for which this.alpha[i] = 0 is irrelevant for future. 
          var newdata = [];
          var newlabels = [];
          var newalpha = [];
          for(var i=0;i<this.N;i++) {
            //console.log("alpha=%f", this.alpha[i]);
            if(this.alpha[i] > alphatol) {
              newdata.push(this.data[i]);
              newlabels.push(this.labels[i]);
              newalpha.push(this.alpha[i]);
            }
          }

          // store data and labels
          this.data = newdata;
          this.labels = newlabels;
          this.alpha = newalpha;
          this.N = this.data.length;
          //console.log("filtered training data from %d to %d support vectors.", data.length, this.data.length);
        }

        var trainstats = {};
        trainstats.iters= iter;
        return trainstats;
      }, 
      
      // inst is an array of length D. Returns margin of given example
      // this is the core prediction function. All others are for convenience mostly
      // and end up calling this one somehow.
      marginOne: function(inst) {

        var f = this.b;
        // if the linear kernel was used and w was computed and stored,
        // (i.e. the svm has fully finished training)
        // the internal class variable usew_ will be set to true.
        if(this.usew_) {

          // we can speed this up a lot by using the computed weights
          // we computed these during train(). This is significantly faster
          // than the version below
          for(var j=0;j<this.D;j++) {
            f += inst[j] * this.w[j];
          }

        } else {

          for(var i=0;i<this.N;i++) {
            f += this.alpha[i] * this.labels[i] * this.kernel(inst, this.data[i]);
          }
        }

        return f;
      },
      
      predictOne: function(inst) { 
        return this.marginOne(inst) > 0 ? 1 : -1; 
      },
      
      // data is an NxD array. Returns array of margins.
      margins: function(data) {
        
        // go over support vectors and accumulate the prediction. 
        var N = data.length;
        var margins = new Array(N);
        for(var i=0;i<N;i++) {
          margins[i] = this.marginOne(data[i]);
        }
        return margins;
        
      },
      
      // data is NxD array. Returns array of 1 or -1, predictions
      predict: function(data) {
        var margs = this.margins(data);
        for(var i=0;i<margs.length;i++) {
          margs[i] = margs[i] > 0 ? 1 : -1;
        }
        return margs;
      },
      
      // THIS FUNCTION IS NOW DEPRECATED. WORKS FINE BUT NO NEED TO USE ANYMORE. 
      // LEAVING IT HERE JUST FOR BACKWARDS COMPATIBILITY FOR A WHILE.
      // if we trained a linear svm, it is possible to calculate just the weights and the offset
      // prediction is then yhat = sign(X * w + b)
      getWeights: function() {
        
        // DEPRECATED
        var w= new Array(this.D);
        for(var j=0;j<this.D;j++) {
          var s= 0.0;
          for(var i=0;i<this.N;i++) {
            s+= this.alpha[i] * this.labels[i] * this.data[i][j];
          }
          w[j]= s;
        }
        return {w: w, b: this.b};
      },

      toJSON: function() {
        
        if(this.kernelType === "custom") {
          console.log("Can't save this SVM because it's using custom, unsupported kernel...");
          return {};
        }

        json = {};
        json.N = this.N;
        json.D = this.D;
        json.b = this.b;

        json.kernelType = this.kernelType;
        if(this.kernelType === "linear") { 
          // just back up the weights
          json.w = this.w; 
        }
        if(this.kernelType === "rbf") { 
          // we need to store the support vectors and the sigma
          json.rbfSigma = this.rbfSigma; 
          json.data = this.data;
          json.labels = this.labels;
          json.alpha = this.alpha;
        }

        return json;
      },
      
      fromJSON: function(json) {
        
        this.N = json.N;
        this.D = json.D;
        this.b = json.b;

        this.kernelType = json.kernelType;
        if(this.kernelType === "linear") { 

          // load the weights! 
          this.w = json.w; 
          this.usew_ = true; 
          this.kernel = linearKernel; // this shouldn't be necessary
        }
        else if(this.kernelType == "rbf") {

          // initialize the kernel
          this.rbfSigma = json.rbfSigma; 
          this.kernel = makeRbfKernel(this.rbfSigma);

          // load the support vectors
          this.data = json.data;
          this.labels = json.labels;
          this.alpha = json.alpha;
        } else {
          console.log("ERROR! unrecognized kernel type." + this.kernelType);
        }
      }
    };
    
    // Kernels
    function makeRbfKernel(sigma) {
      return function(v1, v2) {
        var s=0;
        for(var q=0;q<v1.length;q++) { s += (v1[q] - v2[q])*(v1[q] - v2[q]); } 
        return Math.exp(-s/(2.0*sigma*sigma));
      }
    }
    
    function linearKernel(v1, v2) {
      var s=0; 
      for(var q=0;q<v1.length;q++) { s += v1[q] * v2[q]; } 
      return s;
    }

    // generate random integer between a and b (b excluded)
    function randi(a, b) {
       return Math.floor(Math.random()*(b-a)+a);
    }

    // create vector of zeros of length n
    function zeros(n) {
      var arr= new Array(n);
      for(var i=0;i<n;i++) { arr[i]= 0; }
      return arr;
    }

    // export public members
    exports = exports || {};
    exports.SVM = SVM;
    exports.makeRbfKernel = makeRbfKernel;
    exports.linearKernel = linearKernel;
    return exports;

  }))(module.exports);  // add exports to module.exports if in node.js
  });
  var svm_1 = svm.SVM;

  var SvmJs = function SvmJs(opts) {
    _classCallCheck(this, SvmJs);

    this.base = new svm_1();
    this.opts = opts; // options for SvmJsBase.train
  };

  SvmJs.prototype = {
    trainOnline: function trainOnline(features, label) {
      throw new Error("svm.js does not support online training");
    },
    trainBatch: function trainBatch(dataset) {
      var data = [];
      var labels = [];
      dataset.forEach(function (datum) {
        data.push(datum.input);
        labels.push(datum.output > 0 ? 1 : -1);
      });
      return this.base.train(data, labels, this.opts);
    },

    /**
     * @param features - a feature-value hash.
     * @param explain - int - if positive, an "explanation" field, with the given length, will be added to the result.
     * @param continuous_output if true, return the net classification score. If false [default], return 0 or 1.
     * @return the binary classification - 0 or 1.
     */
    classify: function classify(features, explain, continuous_output) {
      var score = this.base.marginOne(features);
      var classification = continuous_output ? score : score > 0 ? 1 : 0;

      if (explain > 0) {
        this.base.b; // if the linear kernel was used and w was computed and stored,
        // (i.e. the svm has fully finished training)
        // the internal class variable usew_ will be set to true.

        var explanations = [];

        if (this.base.usew_) {
          var w = this.base.w;

          for (var j = 0; j < this.base.D; j++) {
            explanations[j] = {
              feature: j,
              value: features[j],
              weight: w[j],
              relevance: features[j] * w[j]
            };
          }
        }

        explanations.sort(function (a, b) {
          return b.relevance - a.relevance;
        });
        return {
          classification: classification,
          explanation: explanations.slice(0, explain)
        };
      } else {
        return classification;
      }
    },
    toJSON: function toJSON() {
      return this.base.toJSON();
    },
    fromJSON: function fromJSON(json) {
      this.base.fromJSON(json);
    }
  };

  /**
   * Static utilities for hashes (= associative arrays = Javascript objects).
   * 
   * @author Erel Segal-Halevi
   * @since 2013-06
   * @note see performance tests of adding hashes versus arrays here: http://jsperf.com/adding-sparse-feature-vectors
   */
  /**
   * add one hash to another (target += source)
   * @param target [input and output]
   * @param source [input]: will be added to target.
   */

  function add(target, source) {
    for (var feature in source) {
      if (!(feature in target)) target[feature] = 0;
      if (toString.call(target[feature]) != '[object Number]') continue;
      target[feature] += source[feature];
    }

    return target;
  }
  /**
   * multiply a hash by a scalar.
   * @param target [input and output]
   * @param source [input]: target will be multiplied by it.
   */

  function multiply_scalar(target, source) {
    for (var feature in target) {
      if (toString.call(target[feature]) != '[object Number]') continue;
      target[feature] *= source;
    }

    return target;
  }
  function sum_of_absolute_values(weights) {
    var result = 0;

    for (var feature in weights) {
      result += Math.abs(weights[feature]);
    }

    return result;
  }
  /**
   * Normalize the given hash, such that the sum of values is 1.
   * Unless, of course, the current sum is 0, in which case, nothing is done. 
   */

  function normalize_sum_of_values_to_1(features) {
    var sum = sum_of_absolute_values(features);
    if (sum != 0) multiply_scalar(features, 1 / sum);
  }
  /*
  var toStringOrStringArray = function (classes) {
  	if (classes instanceof Array)
  		classes = classes.map(stringifyIfNeeded);
  	else 
  		classes = stringifyIfNeeded(classes);
  	return hash.normalized(classes);
  }
  */

  /**
  sprintf() for JavaScript 0.7-beta1
  http://www.diveintojavascript.com/projects/javascript-sprintf

  Copyright (c) Alexandru Marasteanu <alexaholic [at) gmail (dot] com>
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of sprintf() for JavaScript nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL Alexandru Marasteanu BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


  Changelog:
  2010.11.07 - 0.7-beta1-node
    - converted it to a node.js compatible module

  2010.09.06 - 0.7-beta1
    - features: vsprintf, support for named placeholders
    - enhancements: format cache, reduced global namespace pollution

  2010.05.22 - 0.6:
   - reverted to 0.4 and fixed the bug regarding the sign of the number 0
   Note:
   Thanks to Raphael Pigulla <raph (at] n3rd [dot) org> (http://www.n3rd.org/)
   who warned me about a bug in 0.5, I discovered that the last update was
   a regress. I appologize for that.

  2010.05.09 - 0.5:
   - bug fix: 0 is now preceeded with a + sign
   - bug fix: the sign was not at the right position on padded results (Kamal Abdali)
   - switched from GPL to BSD license

  2007.10.21 - 0.4:
   - unit test and patch (David Baird)

  2007.09.17 - 0.3:
   - bug fix: no longer throws exception on empty paramenters (Hans Pufal)

  2007.09.11 - 0.2:
   - feature: added argument swapping

  2007.04.03 - 0.1:
   - initial release
  **/

  var sprintf = (function() {
  	function get_type(variable) {
  		return Object.prototype.toString.call(variable).slice(8, -1).toLowerCase();
  	}
  	function str_repeat(input, multiplier) {
  		for (var output = []; multiplier > 0; output[--multiplier] = input) {/* do nothing */}
  		return output.join('');
  	}

  	var str_format = function() {
  		if (!str_format.cache.hasOwnProperty(arguments[0])) {
  			str_format.cache[arguments[0]] = str_format.parse(arguments[0]);
  		}
  		return str_format.format.call(null, str_format.cache[arguments[0]], arguments);
  	};

  	// convert object to simple one line string without indentation or
  	// newlines. Note that this implementation does not print array
  	// values to their actual place for sparse arrays. 
  	//
  	// For example sparse array like this
  	//    l = []
  	//    l[4] = 1
  	// Would be printed as "[1]" instead of "[, , , , 1]"
  	// 
  	// If argument 'seen' is not null and array the function will check for 
  	// circular object references from argument.
  	str_format.object_stringify = function(obj, depth, maxdepth, seen) {
  		var str = '';
  		if (obj != null) {
  			switch( typeof(obj) ) {
  			case 'function': 
  				return '[Function' + (obj.name ? ': '+obj.name : '') + ']';
  			case 'object':
  				if ( obj instanceof Error) { return '[' + obj.toString() + ']' }				if (depth >= maxdepth) return '[Object]'
  				if (seen) {
  					// add object to seen list
  					seen = seen.slice(0);
  					seen.push(obj);
  				}
  				if (obj.length != null) { //array
  					str += '[';
  					var arr = [];
  					for (var i in obj) {
  						if (seen && seen.indexOf(obj[i]) >= 0) arr.push('[Circular]');
  						else arr.push(str_format.object_stringify(obj[i], depth+1, maxdepth, seen));
  					}
  					str += arr.join(', ') + ']';
  				} else if ('getMonth' in obj) { // date
  					return 'Date(' + obj + ')';
  				} else { // object
  					str += '{';
  					var arr = [];
  					for (var k in obj) { 
  						if(obj.hasOwnProperty(k)) {
  							if (seen && seen.indexOf(obj[k]) >= 0) arr.push(k + ': [Circular]');
  							else arr.push(k +': ' +str_format.object_stringify(obj[k], depth+1, maxdepth, seen)); 
  						}
  					}
  					str += arr.join(', ') + '}';
  				}
  				return str;
  			case 'string':				
  				return '"' + obj + '"';
  			}
  		}
  		return '' + obj;
  	};

  	str_format.format = function(parse_tree, argv) {
  		var cursor = 1, tree_length = parse_tree.length, node_type = '', arg, output = [], i, k, match, pad, pad_character, pad_length;
  		for (i = 0; i < tree_length; i++) {
  			node_type = get_type(parse_tree[i]);
  			if (node_type === 'string') {
  				output.push(parse_tree[i]);
  			}
  			else if (node_type === 'array') {
  				match = parse_tree[i]; // convenience purposes only
  				if (match[2]) { // keyword argument
  					arg = argv[cursor];
  					for (k = 0; k < match[2].length; k++) {
  						if (!arg.hasOwnProperty(match[2][k])) {
  							throw new Error(sprintf('[sprintf] property "%s" does not exist', match[2][k]));
  						}
  						arg = arg[match[2][k]];
  					}
  				}
  				else if (match[1]) { // positional argument (explicit)
  					arg = argv[match[1]];
  				}
  				else { // positional argument (implicit)
  					arg = argv[cursor++];
  				}

  				if (/[^sO]/.test(match[8]) && (get_type(arg) != 'number')) {
  					throw new Error(sprintf('[sprintf] expecting number but found %s "' + arg + '"', get_type(arg)));
  				}
  				switch (match[8]) {
  					case 'b': arg = arg.toString(2); break;
  					case 'c': arg = String.fromCharCode(arg); break;
  					case 'd': arg = parseInt(arg, 10); break;
  					case 'e': arg = match[7] ? arg.toExponential(match[7]) : arg.toExponential(); break;
  					case 'f': arg = match[7] ? parseFloat(arg).toFixed(match[7]) : parseFloat(arg); break;
  				    case 'O': arg = str_format.object_stringify(arg, 0, parseInt(match[7]) || 5); break;
  					case 'o': arg = arg.toString(8); break;
  					case 's': arg = ((arg = String(arg)) && match[7] ? arg.substring(0, match[7]) : arg); break;
  					case 'u': arg = Math.abs(arg); break;
  					case 'x': arg = arg.toString(16); break;
  					case 'X': arg = arg.toString(16).toUpperCase(); break;
  				}
  				arg = (/[def]/.test(match[8]) && match[3] && arg >= 0 ? '+'+ arg : arg);
  				pad_character = match[4] ? match[4] == '0' ? '0' : match[4].charAt(1) : ' ';
  				pad_length = match[6] - String(arg).length;
  				pad = match[6] ? str_repeat(pad_character, pad_length) : '';
  				output.push(match[5] ? arg + pad : pad + arg);
  			}
  		}
  		return output.join('');
  	};

  	str_format.cache = {};

  	str_format.parse = function(fmt) {
  		var _fmt = fmt, match = [], parse_tree = [], arg_names = 0;
  		while (_fmt) {
  			if ((match = /^[^\x25]+/.exec(_fmt)) !== null) {
  				parse_tree.push(match[0]);
  			}
  			else if ((match = /^\x25{2}/.exec(_fmt)) !== null) {
  				parse_tree.push('%');
  			}
  			else if ((match = /^\x25(?:([1-9]\d*)\$|\(([^\)]+)\))?(\+)?(0|'[^$])?(-)?(\d+)?(?:\.(\d+))?([b-fosOuxX])/.exec(_fmt)) !== null) {
  				if (match[2]) {
  					arg_names |= 1;
  					var field_list = [], replacement_field = match[2], field_match = [];
  					if ((field_match = /^([a-z_][a-z_\d]*)/i.exec(replacement_field)) !== null) {
  						field_list.push(field_match[1]);
  						while ((replacement_field = replacement_field.substring(field_match[0].length)) !== '') {
  							if ((field_match = /^\.([a-z_][a-z_\d]*)/i.exec(replacement_field)) !== null) {
  								field_list.push(field_match[1]);
  							}
  							else if ((field_match = /^\[(\d+)\]/.exec(replacement_field)) !== null) {
  								field_list.push(field_match[1]);
  							}
  							else {
  								throw new Error('[sprintf] ' + replacement_field);
  							}
  						}
  					}
  					else {
                          throw new Error('[sprintf] ' + replacement_field);
  					}
  					match[2] = field_list;
  				}
  				else {
  					arg_names |= 2;
  				}
  				if (arg_names === 3) {
  					throw new Error('[sprintf] mixing positional and named placeholders is not (yet) supported');
  				}
  				parse_tree.push(match);
  			}
  			else {
  				throw new Error('[sprintf] ' + _fmt);
  			}
  			_fmt = _fmt.substring(match[0].length);
  		}
  		return parse_tree;
  	};

  	return str_format;
  })();

  var vsprintf = function(fmt, argv) {
  	var argvClone = argv.slice();
  	argvClone.unshift(fmt);
  	return sprintf.apply(null, argvClone);
  };

  var sprintf_1 = sprintf;
  sprintf.sprintf = sprintf;
  sprintf.vsprintf = vsprintf;
  var sprintf_2 = sprintf_1.sprintf;

  var WinnowHash = function WinnowHash(opts) {
    _classCallCheck(this, WinnowHash);

    if (!opts) opts = {};
    this.debug = opts.debug || false; // Default values are based on Carvalho and Cohen, 2006, section 4.2:	

    this.default_positive_weight = opts.default_positive_weight || 2.0;
    this.default_negative_weight = opts.default_negative_weight || 1.0;
    this.do_averaging = opts.do_averaging || false;
    this.threshold = 'threshold' in opts ? opts.threshold : 1;
    this.promotion = opts.promotion || 1.5;
    this.demotion = opts.demotion || 0.5;
    this.margin = 'margin' in opts ? opts.margin : 1.0;
    this.retrain_count = opts.retrain_count || 0;
    this.detailed_explanations = opts.detailed_explanations || false;
    this.bias = 'bias' in opts ? opts.bias : 1.0;
    this.positive_weights = {};
    this.negative_weights = {};
    this.positive_weights_sum = {}; // for averaging; count only weight vectors with successful predictions (Carvalho and Cohen, 2006).

    this.negative_weights_sum = {}; // for averaging; count only weight vectors with successful predictions (Carvalho and Cohen, 2006).
  };

  WinnowHash.prototype = {
    toJSON: function toJSON(folder) {
      return {
        positive_weights: this.positive_weights,
        negative_weights: this.negative_weights,
        positive_weights_sum: this.positive_weights_sum,
        negative_weights_sum: this.negative_weights_sum
      };
    },
    fromJSON: function fromJSON(json) {
      if (!json.positive_weights) throw new Error("No positive weights in json: " + JSON.stringify(json));
      this.positive_weights = json.positive_weights;
      this.positive_weights_sum = json.positive_weights_sum;
      if (!json.negative_weights) throw new Error("No negative weights in json: " + JSON.stringify(json));
      this.negative_weights = json.negative_weights;
      this.negative_weights_sum = json.negative_weights_sum;
    },
    editFeatureValues: function editFeatureValues(features, remove_unknown_features) {
      if (this.bias && !('bias' in features)) features['bias'] = 1;

      if (remove_unknown_features) {
        for (var feature in features) {
          if (!(feature in this.positive_weights)) delete features[feature];
        }
      }

      normalize_sum_of_values_to_1(features);
    },

    /**
     * @param inputs a SINGLE training sample; a hash (feature => value).
     * @param expected the classification value for that sample (0 or 1)
     * @return true if the input sample got its correct classification (i.e. no change made).
     */
    train_features: function train_features(features, expected) {
      if (this.debug) console.log("train_features " + JSON.stringify(features) + " , " + expected);

      for (feature in features) {
        if (!(feature in this.positive_weights)) this.positive_weights[feature] = this.default_positive_weight;
        if (!(feature in this.negative_weights)) this.negative_weights[feature] = this.default_negative_weight;
      }

      if (this.debug) console.log('> this.positive_weights  ', JSON.stringify(this.positive_weights), ', this.negative_weights: ', JSON.stringify(this.negative_weights));
      var score = this.perceive_features(features,
      /*continuous_output=*/
      true, this.positive_weights, this.negative_weights); // always use the running 'weights' vector for training, and NOT the weights_sum!
      //if (this.debug) console.log('> training ',features,', expecting: ',expected, ' got score=', score);

      if (expected && score <= this.margin || !expected && score >= -this.margin) {
        // Current model is incorrect - adjustment needed!
        if (expected) {
          for (var feature in features) {
            var value = features[feature];
            this.positive_weights[feature] *= this.promotion * (1 + value);
            this.negative_weights[feature] *= this.demotion * (1 - value);
          }
        } else {
          for (var feature in features) {
            var value = features[feature];
            this.positive_weights[feature] *= this.demotion * (1 - value);
            this.negative_weights[feature] *= this.promotion * (1 + value);
          }
        }

        if (this.debug) console.log('--> this.positive_weights', JSON.stringify(this.positive_weights), ', this.negative_weights: ', JSON.stringify(this.negative_weights));
        return false;
      } else {
        if (this.do_averaging) {
          add(this.positive_weights_sum, this.positive_weights);
          add(this.negative_weights_sum, this.negative_weights);
        }

        return true;
      }
    },

    /**
     * train online (a single instance).
     *
     * @param features a SINGLE training sample (a hash of feature-value pairs).
     * @param expected the classification value for that sample (0 or 1).
     * @return true if the input sample got its correct classification (i.e. no change made).
     */
    trainOnline: function trainOnline(features, expected) {
      this.editFeatureValues(features,
      /*remove_unknown_features=*/
      false);
      return this.train_features(features, expected); //this.normalized_features(features, /*remove_unknown_features=*/false), expected);
    },

    /**
     * Batch training (a set of samples). Uses the option this.retrain_count.
     *
     * @param dataset an array of samples of the form {input: {feature1: value1...} , output: 0/1} 
     */
    trainBatch: function trainBatch(dataset) {
      //			var normalized_inputs = [];
      for (var i = 0; i < dataset.length; ++i) {
        this.editFeatureValues(dataset[i].input,
        /*remove_unknown_features=*/
        false);
      } //				normalized_inputs[i] = this.normalized_features(dataset[i].input, /*remove_unknown_features=*/false);


      for (var r = 0; r <= this.retrain_count; ++r) {
        for (var i = 0; i < dataset.length; ++i) {
          this.train_features(dataset[i].input, dataset[i].output);
        }
      }
    },

    /**
     * @param inputs a SINGLE sample; a hash (feature => value).
     * @param continuous_output if true, return the net classification score. If false [default], return 0 or 1.
     * @param explain - int - if positive, an "explanation" field, with the given length, will be added to the result.  
     * @param positive_weights_for_classification, negative_weights_for_classification -
      the weights vector to use (either the running 'weights' or 'weights_sum').  
     * @return the classification of the sample.
     */
    perceive_features: function perceive_features(features, continuous_output, positive_weights_for_classification, negative_weights_for_classification, explain) {
      var score = 0;
      var explanations = [];

      for (var feature in features) {
        if (feature in positive_weights_for_classification) {
          var positive_weight = positive_weights_for_classification[feature];

          if (!isFinite(positive_weight)) {
            console.dir(positive_weights_for_classification);
            throw new Error("positive_weight[" + feature + "]=" + positive_weight);
          }

          var negative_weight = negative_weights_for_classification[feature];

          if (!isFinite(negative_weight)) {
            console.dir(negative_weights_for_classification);
            throw new Error("negative_weight[" + feature + "]=" + negative_weight);
          }

          var net_weight = positive_weight - negative_weight;
          var value = features[feature];

          if (isNaN(value)) {
            console.dir(features);
            throw new Error("score is NaN! features[" + feature + "]=" + value + " net_weight=" + positive_weight + "-" + negative_weight + "=" + net_weight);
          }

          var relevance = value * net_weight;
          score += relevance;
          if (isNaN(score)) throw new Error("score is NaN! features[" + feature + "]=" + value + " net_weight=" + positive_weight + "-" + negative_weight + "=" + net_weight);
          if (explain > 0) explanations.push({
            feature: feature,
            value: value,
            weight: sprintf_2("+%1.3f-%1.3f=%1.3f", positive_weight, negative_weight, net_weight),
            relevance: relevance
          });
        }
      }

      if (isNaN(score)) throw new Error("score is NaN! features=" + JSON.stringify(features));
      score -= this.threshold;
      if (this.debug) console.log("> perceive_features ", JSON.stringify(features), " = ", score);
      var result = continuous_output ? score : score > 0 ? 1 : 0;

      if (explain > 0) {
        explanations.sort(function (a, b) {
          return Math.abs(b.relevance) - Math.abs(a.relevance);
        });
        explanations.splice(explain, explanations.length - explain); // "explain" is the max length of explanation.

        if (!this.detailed_explanations) {
          explanations = explanations.map(function (e) {
            return sprintf_2("%s%+1.2f", e.feature, e.relevance);
          });
        }

        result = {
          classification: result,
          explanation: explanations
        };
      }

      return result;
    },

    /**
     * @param inputs a SINGLE sample (a hash of feature-value pairs).
     * @param continuous_output if true, return the net classification value. If false [default], return 0 or 1.
     * @param explain - int - if positive, an "explanation" field, with the given length, will be added to the result.  
     * @return the classification of the sample.
     */
    classify: function classify(features, explain, continuous_output) {
      this.editFeatureValues(features,
      /*remove_unknown_features=*/
      true);
      return this.perceive_features( //this.normalized_features(features, /*remove_unknown_features=*/true),
      features, continuous_output, this.do_averaging ? this.positive_weights_sum : this.positive_weights, this.do_averaging ? this.negative_weights_sum : this.negative_weights, explain);
    }
  };

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
      SvmJs: SvmJs,
      SvmLinear: SvmLinear,
      SvmPerf: SvmPerf,
      Winnow: WinnowHash,
      multilabel: multilabel,
      EnhancedClassifier: EnhancedClassifier
    },
    features: features,
    formats: formats,
    utils: utils
  };

})));
