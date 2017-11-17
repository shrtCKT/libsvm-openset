from svm import *
import numpy as np
from ctypes import *


class PiSVM():
    def __init__(self, gamma=None, p_threshold=0.25, contamination=None):
        """
        Args:
        gamma - set gamma in kernel function (if None then set to 1/num_features)
        p_threshold - probability threshold value to reject sample as unknowns for
                      (default 0.25)
        """
        self.param = svm_parameter()
        self.param.set_to_default_values()
        self.param.print_func = cast(None, PRINT_STRING_FUN)

        self.param.svm_type = PI_SVM    # enum constants defined in svm.py
        self.param.kernel_type = RBF
        self.param.openset_min_probability = p_threshold
        self.p_threshold = p_threshold
        self.svm_model = None
        self.contamination = contamination

    @staticmethod
    def to_svm_problem(X, y):
        """ Converts numpy format to svm problem format

        Args:
        X - 2d numpy array of feature vectors
        y - 1d numpy array

        Returns:
        svm_problem
        """
        prob_y = []
        prob_x = []
        for i in xrange(X.shape[0]):
            xi = {}
            for ind in xrange(X.shape[1]):
                xi[int(ind)] = float(X[i, ind])
            prob_y += [float(y[i])]
            prob_x += [xi]
        return svm_problem(prob_y, prob_x)

    @staticmethod
    def to_svm_problem_x(X):
        """ Converts numpy format to svm problem format

        Args:
        X - 2d numpy array of feature vectors

        Returns:
        problem_x
        """
        prob_x = []
        for i in xrange(X.shape[0]):
            xi = {}
            for ind in xrange(X.shape[1]):
                xi[int(ind)] = float(X[i, ind])
            prob_x += [xi]
        return prob_x

    def fit(self, X, y):
        """ Train a PI_SVM model

        Args:
        X - 2d numpy array of feature vectors
        y - 1d numpy array
        """
        prob = PiSVM.to_svm_problem(X, y)
        if self.param.gamma == 0 and prob.n > 0: 
            self.param.gamma = 1.0 / prob.n
        libsvm.svm_set_print_string_function(self.param.print_func)
        err_msg = libsvm.svm_check_parameter(prob, self.param)
        if err_msg:
            raise ValueError('Error: %s' % err_msg)

        self.svm_model = libsvm.svm_train(prob, self.param)
        self.svm_model = toPyModel(self.svm_model)

        # If prob is destroyed, data including SVs pointed by m can remain.
        self.svm_model.x_space = prob.x_space

    def _predict(self, X):
        """ For internal use only
        """
        svm_type = self.svm_model.get_svm_type()
        is_prob_model = self.svm_model.is_probability_model()
        nr_class = self.svm_model.get_nr_class()
        pred_labels = []
        pred_max_score = []

        x = PiSVM.to_svm_problem_x(X)
        for xi in x:
            # create score and vote arrays to be passed to libsvm.svm_predict_extended
            l = [[0.0] * nr_class] * (nr_class + 1)
            entrylist = []
            for sub_l in l:
                entrylist.append((c_double*len(sub_l))(*sub_l))
            scores = (POINTER(c_double) * len(entrylist))(*entrylist)
            votes = (c_int * (nr_class + 1))()

            # make prediction
            xi, idx = gen_svm_nodearray(xi)
            label = libsvm.svm_predict_extended(self.svm_model, xi,
                                                byref(cast(scores, POINTER(POINTER(c_double)))),
                                                byref(cast(votes, POINTER(c_int))))
            pred_labels += [label]
            max_prob = scores[0][0]
            for jj in xrange(self.svm_model.openset_dim):
                if scores[jj][0] > max_prob:
                    max_prob = scores[jj][0]

            pred_max_score += [max_prob]

        return pred_labels, pred_max_score

    def predict(self, X):
        """ Predict class of instances in X.
        If an instances determined to be an outlier then predicted as class -1

        Args:
        X - 2d numpy array
        
        Returns:
        1D numpy array of class predictions. If an instances determined to be 
        an outlier then predicted as class -1 other wise the predicted class. 
        """
        pred_labels, pred_score = self._predict(X)
        y_score = np.array(pred_score)
        y_pred = np.array(pred_labels, dtype='int')
        y_pred[y_score < self.p_threshold] = -1
        return y_pred

    def decision_function(self, X):
        """ Predicted score.
        """
        pred_labels, pred_score = self._predict(X)
        return np.array(pred_score)

    