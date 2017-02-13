#!/usr/bin/env python

from svm import *

def svm_read_problem(data_file_name):
	"""
	svm_read_problem(data_file_name) -> [y, x]

	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	prob_y = []
	prob_x = []
	for line in open(data_file_name):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		xi = {}
		for e in features.split():
			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		prob_y += [float(label)]
		prob_x += [xi]
	return (prob_y, prob_x)

def svm_load_model(model_file_name):
	"""
	svm_load_model(model_file_name) -> model
	
	Load a LIBSVM model from model_file_name and return.
	"""
	model = libsvm.svm_load_model(model_file_name)
	if not model: 
		print("can't open model file %s" % model_file_name)
		return None
	model = toPyModel(model)
	return model

def svm_save_model(model_file_name, model):
	"""
	svm_save_model(model_file_name, model) -> None

	Save a LIBSVM model to the file model_file_name.
	"""
	libsvm.svm_save_model(model_file_name, model)

def evaluations(ty, pv):
	"""
	evaluations(ty, pv) -> (ACC, MSE, SCC)

	Calculate accuracy, mean squared error and squared correlation coefficient
	using the true values (ty) and predicted values (pv).
	"""
	if len(ty) != len(pv):
		raise ValueError("len(ty) must equal to len(pv)")
	total_correct = total_error = 0
	sumv = sumy = sumvv = sumyy = sumvy = 0
	for v, y in zip(pv, ty):
		if y == v: 
			total_correct += 1
		total_error += (v-y)*(v-y)
		sumv += v
		sumy += y
		sumvv += v*v
		sumyy += y*y
		sumvy += v*y 
	l = len(ty)
	ACC = 100.0*total_correct/l
	MSE = total_error/l
	try:
		SCC = ((l*sumvy-sumv*sumy)*(l*sumvy-sumv*sumy))/((l*sumvv-sumv*sumv)*(l*sumyy-sumy*sumy))
	except:
		SCC = float('nan')
	return (ACC, MSE, SCC)

def svm_train(arg1, arg2=None, arg3=None):
	"""
	svm_train(y, x [, 'options']) -> model | ACC | MSE 
	svm_train(prob, [, 'options']) -> model | ACC | MSE 
	svm_train(prob, param) -> model | ACC| MSE 

	Train an SVM model from data (y, x) or an svm_problem prob using
	'options' or an svm_parameter param. 
	If '-v' is specified in 'options' (i.e., cross validation)
	either accuracy (ACC) or mean-squared error (MSE) is returned.
	'options':
	    -s svm_type : set type of SVM (default 0)
	        0 -- C-SVC
	        1 -- nu-SVC
	        2 -- one-class SVM
	        3 -- epsilon-SVR
	        4 -- nu-SVR
			5 -- open-set oneclass SVM (open_set_training_file required)
			6 -- open-set pair-wise SVM  (open_set_training_file required)
			7 -- open-set binary SVM  (open_set_training_file required)
			8 -- one-vs-rest WSVM (open_set_training_file required)
			9 -- One-class PI-OSVM (open_set_training_file required)
			10 -- one-vs-all PI-SVM (open_set_training_file required) 
	    -t kernel_type : set type of kernel function (default 2)
	        0 -- linear: u'*v
	        1 -- polynomial: (gamma*u'*v + coef0)^degree
	        2 -- radial basis function: exp(-gamma*|u-v|^2)
	        3 -- sigmoid: tanh(gamma*u'*v + coef0)
	        4 -- precomputed kernel (kernel values in training_set_file)
	    -d degree : set degree in kernel function (default 3)
	    -g gamma : set gamma in kernel function (default 1/num_features)
	    -r coef0 : set coef0 in kernel function (default 0)
	    -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
	    -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
	    -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
	    -m cachesize : set cache memory size in MB (default 100)
	    -e epsilon : set tolerance of termination criterion (default 0.001)
	    -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
	    -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
	    -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
	    -v n: n-fold cross validation mode
	    -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		-q : quiet mode (no outputs)
		-P threshold : probability value to reject sample as unknowns for WSVM/One-class PI-OSVM (default 0.0) (only for cross validation)
        -C threshold : probability value to reject sample as unknowns for CAP model in WSVM(default 0.0) (only for cross validation)
		-B beta : will set the beta for fmeasure used in openset training, default =1
		-N : we build models for negative classes (used for multiclass where labels might be negative.  default is only positive models. (default is false)
		-E : do exaustive search for best openset (otherwise do the default greedy optimization). (default is false)
		-G nearpreasure farpressure : will adjust the pressures for openset optimiation. <0 will specalize, >0 will generalize
		-o cost : set the parameter C for CAP model in one-vs-rest WSVM
        -a gamma : set gamma in kernel function for CAP model in one-vs-rest WSVM
	"""
	prob, param = None, None
	if isinstance(arg1, (list, tuple)):
		assert isinstance(arg2, (list, tuple))
		y, x, options = arg1, arg2, arg3
		prob = svm_problem(y, x)
		param = svm_parameter(options)
	elif isinstance(arg1, svm_problem):
		prob = arg1
		if isinstance(arg2, svm_parameter):
			param = arg2
		else:
			param = svm_parameter(arg2)
	if prob == None or param == None:
		raise TypeError("Wrong types for the arguments")

	if param.kernel_type == PRECOMPUTED:
		for xi in prob.x_space:
			idx, val = xi[0].index, xi[0].value
			if xi[0].index != 0:
				raise ValueError('Wrong input format: first column must be 0:sample_serial_number')
			if val <= 0 or val > prob.n:
				raise ValueError('Wrong input format: sample_serial_number out of range')

	if param.gamma == 0 and prob.n > 0: 
		param.gamma = 1.0 / prob.n
	libsvm.svm_set_print_string_function(param.print_func)
	err_msg = libsvm.svm_check_parameter(prob, param)
	if err_msg:
		raise ValueError('Error: %s' % err_msg)

	if param.cross_validation:
		l, nr_fold = prob.l, param.nr_fold
		target = (c_double * l)()
		libsvm.svm_cross_validation(prob, param, nr_fold, target)	
		ACC, MSE, SCC = evaluations(prob.y[:l], target[:l])
		if param.svm_type in [EPSILON_SVR, NU_SVR]:
			print("Cross Validation Mean squared error = %g" % MSE)
			print("Cross Validation Squared correlation coefficient = %g" % SCC)
			return MSE
		else:
			print("Cross Validation Accuracy = %g%%" % ACC)
			return ACC
	else:
		m = libsvm.svm_train(prob, param)
		m = toPyModel(m)

		# If prob is destroyed, data including SVs pointed by m can remain.
		m.x_space = prob.x_space
		return m

def svm_predict(y, x, m, options=""):
	"""
	svm_predict(y, x, m [, "options"]) -> (p_labels, p_acc, p_vals)

	Predict data (y, x) with the SVM model m. 
	"options": 
	    -b probability_estimates: whether to predict probability estimates, 
	        0 or 1 (default 0); for one-class SVM only 0 is supported.

	The return tuple contains
	p_labels: a list of predicted labels
	p_acc: a tuple including  accuracy (for classification), mean-squared 
	       error, and squared correlation coefficient (for regression).
	p_vals: a list of decision values or probability estimates (if '-b 1' 
	        is specified). If k is the number of classes, for decision values,
	        each element includes results of predicting k(k-1)/2 binary-class
	        SVMs. For probabilities, each element contains k values indicating
	        the probability that the testing instance is in each class.
	        Note that the order of classes here is the same as 'model.label'
	        field in the model structure.
	"""
	predict_probability = 0
	argv = options.split()
	i = 0
	while i < len(argv):
		if argv[i] == '-b':
			i += 1
			predict_probability = int(argv[i])
		else:
			raise ValueError("Wrong options")
		i+=1

	svm_type = m.get_svm_type()
	is_prob_model = m.is_probability_model()
	nr_class = m.get_nr_class()
	pred_labels = []
	pred_values = []
	pred_max_score = []

	if predict_probability:
		if not is_prob_model:
			raise ValueError("Model does not support probabiliy estimates")

		if svm_type in [NU_SVR, EPSILON_SVR]:
			print("Prob. model for test data: target value = predicted value + z,\n"
			"z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g" % m.get_svr_probability());
			nr_class = 0

		prob_estimates = (c_double * nr_class)()
		for xi in x:
			xi, idx = gen_svm_nodearray(xi)
			label = libsvm.svm_predict_probability(m, xi, prob_estimates)
			values = prob_estimates[:nr_class]
			pred_labels += [label]
			pred_values += [values]
	elif svm_type == ONE_WSVM or svm_type == PI_SVM:
		# votes = (c_int * (nr_class + 1))()
		# scores = ((c_double * nr_class) * (nr_class + 1))()
		
		
		l = [[0.0] * nr_class] * (nr_class + 1)
		entrylist = []
		for sub_l in l:
			entrylist.append((c_double*len(sub_l))(*sub_l))
		scores = (POINTER(c_double) * len(entrylist))(*entrylist)
		votes = (c_int * (nr_class + 1))()

		for xi in x:
			xi, idx = gen_svm_nodearray(xi)
			label = libsvm.svm_predict_extended(m, xi, 
												byref(cast(scores, POINTER(POINTER(c_double)))),
												byref(cast(votes, POINTER(c_int))))
			pred_labels += [label]
			max_prob = scores[0][0];
			for jj in xrange(m.openset_dim):
				if(scores[jj][0] > max_prob):
					max_prob = scores[jj][0]
			
			pred_max_score += [max_prob]
	else:
		if is_prob_model:
			print("Model supports probability estimates, but disabled in predicton.")
		if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
			nr_classifier = 1
		else:
			nr_classifier = nr_class*(nr_class-1)//2
		dec_values = (c_double * nr_classifier)()
		for xi in x:
			xi, idx = gen_svm_nodearray(xi)
			label = libsvm.svm_predict_values(m, xi, dec_values)
			values = dec_values[:nr_classifier]
			pred_labels += [label]
			pred_values += [values]

	ACC, MSE, SCC = evaluations(y, pred_labels)
	l = len(y)
	if svm_type in [EPSILON_SVR, NU_SVR]:
		print("Mean squared error = %g (regression)" % MSE)
		print("Squared correlation coefficient = %g (regression)" % SCC)
	else:
		print("Accuracy = %g%% (%d/%d) (classification)" % (ACC, int(l*ACC/100), l))

	return pred_labels, (ACC, MSE, SCC), pred_values, pred_max_score

