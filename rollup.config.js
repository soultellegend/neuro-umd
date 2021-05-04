import babel from 'rollup-plugin-babel';
import resolve from 'rollup-plugin-node-resolve';
import commonjs from 'rollup-plugin-commonjs';

export default [{
	input: './src/index.js',
	output: {
		file: './dist/index.umd.js',
		format: 'umd',
		sourceMap: 'inline',
	},
	plugins: [
		resolve({
			mainFields: ['module', 'main', 'jsnext:main', 'browser'],
			extensions: [ '.js' ],
		}),
		commonjs({
			include: /node_modules/,
			namedExports: {
				'underscore': [	'reduce', 'isObject', 'isArray', 'each', 'clone' ],
				'lodash': [ 'isArray', 'forEach', 'compact', 'flattenDeep', 'map' ],
				'svm': [ 'SVM' ],
				'sprintf': [ 'sprintf' ],
			},
		}),
		babel({
			exclude: './node_modules/**',
			extensions: [ '.js' ],
		}),
	],
}]
