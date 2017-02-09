#!/usr/bin/env python

from ctypes import *
from ctypes.util import find_library
import sys
import os

# For unix the prefix 'lib' is not considered.
if find_library('svm'):
	libsvm = CDLL(find_library('svm'))
elif find_library('libsvm'):
	libsvm = CDLL(find_library('libsvm'))
else:
	if sys.platform == 'win32':
		libsvm = CDLL(os.path.join(os.path.dirname(__file__),\
				'../windows/libsvm.dll'))
	else:
		libMR = CDLL(os.path.join(os.path.dirname(__file__),\
				'../libMR/build/libMR/libMR.so'))
		libsvm = CDLL(os.path.join(os.path.dirname(__file__),\
				'../libsvm.so.2'))

# Construct constants
SVM_TYPE = ['C_SVC', 'NU_SVC', 'ONE_CLASS', 'EPSILON_SVR', 'NU_SVR',
			'OPENSET_OC', 'OPENSET_PAIR', 'OPENSET_BIN', 'ONE_VS_REST_WSVM',
			'ONE_WSVM', 'PI_SVM']
KERNEL_TYPE = ['LINEAR', 'POLY', 'RBF', 'SIGMOID', 'PRECOMPUTED']
OPENSET_OPTIMIZATION = ['OPT_PRECISION', 'OPT_RECALL',  'OPT_FMEASURE',  'OPT_HINGE', 'OPT_BALANCEDRISK']
for i, s in enumerate(SVM_TYPE): exec("%s = %d" % (s , i))
for i, s in enumerate(KERNEL_TYPE): exec("%s = %d" % (s , i))
for i, s in enumerate(OPENSET_OPTIMIZATION): exec("%s = %d" % (s , i))

PRINT_STRING_FUN = CFUNCTYPE(None, c_char_p)
def print_null(s): 
	return 

def genFields(names, types): 
	return list(zip(names, types))

def fillprototype(f, restype, argtypes): 
	f.restype = restype
	f.argtypes = argtypes

class svm_node(Structure):
	_names = ["index", "value"]
	_types = [c_int, c_double]
	_fields_ = genFields(_names, _types)

def gen_svm_nodearray(xi, feature_max=None, issparse=None):
	if isinstance(xi, dict):
		index_range = xi.keys()
	elif isinstance(xi, (list, tuple)):
		index_range = range(len(xi))
	else:
		raise TypeError('xi should be a dictionary, list or tuple')

	if feature_max:
		assert(isinstance(feature_max, int))
		index_range = filter(lambda j: j <= feature_max, index_range)
	if issparse: 
		index_range = filter(lambda j:xi[j] != 0, index_range)

	index_range = sorted(index_range)
	ret = (svm_node * (len(index_range)+1))()
	ret[-1].index = -1
	for idx, j in enumerate(index_range):
		ret[idx].index = j
		ret[idx].value = xi[j]
	max_idx = 0
	if index_range: 
		max_idx = index_range[-1]
	return ret, max_idx

class svm_problem(Structure):
	_names = ["l", "y", "x", "nr_classes", "labels"]
	_types = [c_int, POINTER(c_double), POINTER(POINTER(svm_node)), c_int, c_int]
	_fields_ = genFields(_names, _types)

	def __init__(self, y, x):
		if len(y) != len(x):
			raise ValueError("len(y) != len(x)")
		self.l = l = len(y)

		max_idx = 0
		x_space = self.x_space = []
		for i, xi in enumerate(x):
			tmp_xi, tmp_idx = gen_svm_nodearray(xi)
			x_space += [tmp_xi]
			max_idx = max(max_idx, tmp_idx)
		self.n = max_idx

		self.y = (c_double * l)()
		for i, yi in enumerate(y): self.y[i] = yi

		self.x = (POINTER(svm_node) * l)() 
		for i, xi in enumerate(self.x_space): self.x[i] = xi

class svm_parameter(Structure):
	_names = ["svm_type", "kernel_type", "do_open", "degree", "gamma", "coef0",
			"cache_size", "eps", "C", "nr_weight", "nr_fold", "cross_validation", "weight_label", "weight",
			"nu", "p", "shrinking", "probability",
			"neg_labels", "exaustive_open", "optimize", "beta",
			"near_preasure", "far_preasure", "openset_min_probability",
			"openset_min_probability_one_wsvm", "vfile",
			"rejectedID", "cap_cost", "cap_gamma"]
	_types = [c_int, c_int, c_int, c_int, c_double, c_double, # "svm_type", "kernel_type", "do_open", "degree", "gamma", "coef0",
			c_double, c_double, c_double, c_int, c_int, c_int, POINTER(c_int), POINTER(c_double), #"cache_size", "eps", "C", "nr_weight", "nr_fold", "cross_validation", "weight_label", "weight",
			c_double, c_double, c_int, c_int,	# "nu", "p", "shrinking", "probability",
			c_bool, c_bool, c_int, c_double,	# "neg_labels", "exaustive_open", "optimize", "beta",
			c_double, c_double, c_double,		# "near_preasure", "far_preasure", "openset_min_probability",
			c_double, c_void_p, 				# "openset_min_probability_one_wsvm", "vfile",
			c_int, c_double, c_double			# "rejectedID", "cap_cost", "cap_gamma"
			]
	_fields_ = genFields(_names, _types)

	def __init__(self, options = None):
		if options == None:
			options = ''
		self.parse_options(options)

	def show(self):
		attrs = svm_parameter._names + self.__dict__.keys()
		values = map(lambda attr: getattr(self, attr), attrs) 
		for attr, val in zip(attrs, values):
			print(' %s: %s' % (attr, val))

	def set_to_default_values(self):
		self.svm_type = C_SVC;
		self.kernel_type = RBF
		self.degree = 3
		self.gamma = 0
		self.coef0 = 0
		self.nu = 0.5
		self.cache_size = 100
		self.C = 1
		self.eps = 0.001
		self.p = 0.1
		self.shrinking = 1
		self.probability = 0
		self.nr_weight = 0
		self.weight_label = (c_int*0)()
		self.weight = (c_double*0)()
		self.cross_validation = False
		self.do_open = 0
		self.openset_min_probability = 0.0
		self.openset_min_probability_one_wsvm = 0.0
		self.nr_fold = 0
		self.print_func = None

		self.optimize = OPT_BALANCEDRISK
		self.beta = 1.000 # Require classic fmeasure balance of recall and precision by default
		self.near_preasure = 0
		self.far_preasure = 0
		self.rejectedID = -99999
		self.neg_labels = False
		self.exaustive_open = False

	def parse_options(self, options):
		argv = options.split()
		self.set_to_default_values()
		self.print_func = cast(None, PRINT_STRING_FUN)
		weight_label = []
		weight = []

		i = 0
		while i < len(argv):
			if argv[i] == "-s":
				i = i + 1
				self.svm_type = int(argv[i])
			elif argv[i] == "-t":
				i = i + 1
				self.kernel_type = int(argv[i])
			elif argv[i] == "-d":
				i = i + 1
				self.degree = int(argv[i])
			elif argv[i] == "-g":
				i = i + 1
				self.gamma = float(argv[i])
			elif argv[i] == "-r":
				i = i + 1
				self.coef0 = float(argv[i])
			elif argv[i] == "-n":
				i = i + 1
				self.nu = float(argv[i])
			elif argv[i] == "-m":
				i = i + 1
				self.cache_size = float(argv[i])
			elif argv[i] == "-c":
				i = i + 1
				self.C = float(argv[i])
			elif argv[i] == "-e":
				i = i + 1
				self.eps = float(argv[i])
			elif argv[i] == "-p":
				i = i + 1
				self.p = float(argv[i])
			elif argv[i] == "-h":
				i = i + 1
				self.shrinking = int(argv[i])
			elif argv[i] == "-b":
				i = i + 1
				self.probability = int(argv[i])
			elif argv[i] == "-q":
				self.print_func = PRINT_STRING_FUN(print_null)
			elif argv[i] == "-v":
				i = i + 1
				self.cross_validation = 1
				self.nr_fold = int(argv[i])
				if self.nr_fold < 2:
					raise ValueError("n-fold cross validation: n must >= 2")
			elif argv[i].startswith("-w"):
				i = i + 1
				self.nr_weight += 1
				nr_weight = self.nr_weight
				weight_label += [int(argv[i-1][2:])]
				weight += [float(argv[i])]
			else:
				raise ValueError("Wrong options")
			i += 1

		libsvm.svm_set_print_string_function(self.print_func)
		self.weight_label = (c_int*self.nr_weight)()
		self.weight = (c_double*self.nr_weight)()
		for i in range(self.nr_weight): 
			self.weight[i] = weight[i]
			self.weight_label[i] = weight_label[i]

class svm_model(Structure):
	_names = ['param', 'nr_class', 'l', 'SV', 'sv_coef', 'rho',
			'probA', 'probB', 'label', 'nSV', 'free_sv']
	_types = [svm_parameter, c_int, c_int, POINTER(POINTER(svm_node)),
			POINTER(POINTER(c_double)), POINTER(c_double),
			POINTER(c_double), POINTER(c_double), POINTER(c_int),
			POINTER(c_int), c_int]
	_fields_ = genFields(_names, _types)

	def __init__(self):
		self.__createfrom__ = 'python'

	def __del__(self):
		# free memory created by C to avoid memory leak
		if hasattr(self, '__createfrom__') and self.__createfrom__ == 'C':
			libsvm.svm_free_and_destroy_model(pointer(self))

	def get_svm_type(self):
		return libsvm.svm_get_svm_type(self)

	def get_nr_class(self):
		return libsvm.svm_get_nr_class(self)

	def get_svr_probability(self):
		return libsvm.svm_get_svr_probability(self)

	def get_labels(self):
		nr_class = self.get_nr_class()
		labels = (c_int * nr_class)()
		libsvm.svm_get_labels(self, labels)
		return labels[:nr_class]

	def is_probability_model(self):
		return (libsvm.svm_check_probability_model(self) == 1)

	def get_sv_coef(self):
		return [tuple(self.sv_coef[j][i] for j in xrange(self.nr_class - 1))
				for i in xrange(self.l)]

	def get_SV(self):
		result = []
		for sparse_sv in self.SV[:self.l]:
			row = dict()
			
			i = 0
			while True:
				row[sparse_sv[i].index] = sparse_sv[i].value
				if sparse_sv[i].index == -1:
					break
				i += 1

			result.append(row)
		return result

def toPyModel(model_ptr):
	"""
	toPyModel(model_ptr) -> svm_model

	Convert a ctypes POINTER(svm_model) to a Python svm_model
	"""
	if bool(model_ptr) == False:
		raise ValueError("Null pointer")
	m = model_ptr.contents
	m.__createfrom__ = 'C'
	return m

fillprototype(libsvm.svm_train, POINTER(svm_model), [POINTER(svm_problem), POINTER(svm_parameter)])
fillprototype(libsvm.svm_cross_validation, None, [POINTER(svm_problem), POINTER(svm_parameter), c_int, POINTER(c_double)])

fillprototype(libsvm.svm_save_model, c_int, [c_char_p, POINTER(svm_model)])
fillprototype(libsvm.svm_load_model, POINTER(svm_model), [c_char_p])

fillprototype(libsvm.svm_get_svm_type, c_int, [POINTER(svm_model)])
fillprototype(libsvm.svm_get_nr_class, c_int, [POINTER(svm_model)])
fillprototype(libsvm.svm_get_labels, None, [POINTER(svm_model), POINTER(c_int)])
fillprototype(libsvm.svm_get_svr_probability, c_double, [POINTER(svm_model)])

fillprototype(libsvm.svm_predict_values, c_double, [POINTER(svm_model), POINTER(svm_node), POINTER(c_double)])
fillprototype(libsvm.svm_predict, c_double, [POINTER(svm_model), POINTER(svm_node)])
fillprototype(libsvm.svm_predict_probability, c_double, [POINTER(svm_model), POINTER(svm_node), POINTER(c_double)])

fillprototype(libsvm.svm_free_model_content, None, [POINTER(svm_model)])
fillprototype(libsvm.svm_free_and_destroy_model, None, [POINTER(POINTER(svm_model))])
fillprototype(libsvm.svm_destroy_param, None, [POINTER(svm_parameter)])

fillprototype(libsvm.svm_check_parameter, c_char_p, [POINTER(svm_problem), POINTER(svm_parameter)])
fillprototype(libsvm.svm_check_probability_model, c_int, [POINTER(svm_model)])
fillprototype(libsvm.svm_set_print_string_function, None, [PRINT_STRING_FUN])
