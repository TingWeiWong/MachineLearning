# Reference: Google Machine Learning Recipe 
import time 

def load_data(data_path):
	"""
	This function loads training and testing data from HW6 site.
	- Input:
		* data_path: relative path of data
	- Output:
		* two_D_matrix: A two dimensional matrix that holds features and labels
			with label being at the last column
	"""
	
	content_list = []
	
	with open(data_path) as read_file:
		for line in read_file:
			row = []
			line = line.rstrip('\n')
			row = [float (ele) for ele in line.split(" ")]
			content_list.append(row)

	return content_list
	
def analyze_class(training_data):
	"""
	Counts the number unique examples in a dataset.
	This is used to balance the gini_index
	- Output:
		* counts: dictionary with key = 1.0 or -1.0, 
				  values = accumulative counts
	"""
	counts = {}
	for row in training_data:
		label = row[-1]
		if label not in counts:
			counts[label] = 0
		counts[label] += 1
	return counts

def compute_impurity(training_data):
	"""
	Calculate the Gini Impurity.
	1 - Sigma((label) / N)
	"""
	counts = analyze_class(training_data)
	data_num = len(training_data)
	impurity = 1

	for label in counts:
		prob_of_label = counts[label] / float(data_num)
		impurity -= prob_of_label**2
	
	return impurity

def gain_difference(left, right, root_impurity):
	"""
	This function calculates the information gain difference by
	balancing left child and right child 
	- Input:
		* left, right : children
		* root_impurity
	"""
	balancing_factor = float(len(left)) / (len(left) + len(right))
	updated_impurity = root_impurity - balancing_factor * compute_impurity(left) - (1 - balancing_factor) * compute_impurity(right)
	
	return updated_impurity

class Criterion:
	"""
	This class holds a checking creterion for 
	using a specific threshold and column(feature)
	- Input:
		* feature: column name or index
		* value: threshold for comparison

	- Function:
		* match: check if the contesting data matches with this 
			criterion.
	"""
	def __init__(self, feature, value):
		self.feature = feature
		self.value = value

	def match(self, example):
		contest_value = example[self.feature]
		return contest_value >= self.value


def partition(training_data, criterion):
	"""
	This function splits the dataset into true and false
	lists with .
	- Input:
		* training_data
		* criterion: checking if the result should match
	"""
	left_child, right_child = [], []
	for row in training_data:
		if criterion.match(row):
			# print ("Left child matches!")
			left_child.append(row)
		else:
			right_child.append(row)
	return left_child, right_child


def stump_values(raw_values_set):
	"""
	This function uses the decision stump algo to find the N-1 points 
	in unique values
	- Input:
		* raw_values_set: set of unsorted unique values
	- Output:
		* decision_list: list of sorted values using decision stump
	"""
	sorted_list = sorted(raw_values_set)
	length = len(sorted_list)
	decision_list = []

	for i in range(length-1):
		first, second = sorted_list[i], sorted_list[i+1]
		middle = (first + second) / 2.0
		decision_list.append(middle)

	return decision_list

def find_best_split(input_data):
	"""
	This function finds the best criterion to hold given datasets
	- Input:
		* input_data: rows of input data
	"""
	best_gain, best_criterion = 0, None
	root_impurity = compute_impurity(input_data)
	column_number = len(input_data[0]) - 1

	for col in range(column_number):

		values = set([row[col] for row in input_data])

		decision_list = stump_values(values)

		for val in decision_list:
			criterion = Criterion(col, val)
			left_child, right_child = partition(input_data, criterion)

			if len(left_child) == 0 or len(right_child) == 0:
				continue

			gain = gain_difference(left_child, right_child, root_impurity)
			if gain >= best_gain:
				best_gain, best_criterion = gain, criterion

	return best_gain, best_criterion

class BaseNode:
	"""
	A BaseNode node classifies data.
	This basenode holds a dictionary of the feature that the number of times 
	it reaches this node.
	"""
	def __init__(self, rows):
		self.predictions = analyze_class(rows)


class RecursiveNode:
	"""
	A Decision Node asks a criterion.
	This is the recursive Node, holding two children and its criterion.
	"""

	def __init__(self,criterion,left_side,right_side):
		self.criterion = criterion
		self.left_side = left_side
		self.right_side = right_side

def Construction(input_data):
	"""
	This is the function for constructing the tree.
	- Input:
		* input_data
	"""

	start_time = time.time()
	gain, criterion = find_best_split(input_data)
	print("--- %s seconds for find_best_split---" % (time.time() - start_time))	

	if gain == 0:
		return BaseNode(input_data)

	start_time = time.time()
	left_child, right_child = partition(input_data, criterion)
	print("--- %s seconds for partition---" % (time.time() - start_time))	

	start_time = time.time()	
	left_side = Construction(left_child)
	right_side = Construction(right_child)
	print("--- %s seconds for Recursive calls---" % (time.time() - start_time))	

	return RecursiveNode(criterion, left_side, right_side)



def Sort_data(single_data, node):
	"""
	This function classifies the single_data after 
	passing through the criterion.
	"""
	if isinstance(node, BaseNode):
		return node.predictions

	if node.criterion.match(single_data):
		return Sort_data(single_data, node.left_side)
	else:
		return Sort_data(single_data, node.right_side)

def get_label(counts):
	for label in counts.keys():
		result = label
	return result




if __name__ == "__main__":

	mode = 0

	if mode == 0:
		training_data = load_data("data/hw6_train.dat")
		testing_data = load_data("data/hw6_test.dat")
	else:
		training_data = [
			[1.02, 3, 1.0],
			[0.529, 3, 1.0],
			[0.827, 1, -1.0],
			[0.827, 1, -1.0],
			[0.529, 3, 1.0],
		]

		testing_data = [
			[1.02, 3, 1.0],
			[0.529, 4, 1.0],
			[0.827, 2, -1.0],
			[0.827, 1, -1.0],
			[0.529, 3, 1.0],
		]

	counts = analyze_class(training_data)
	header = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"]
	# a = Criterion(1, 3)
	# print (a)
	start_time = time.time()
	my_tree = Construction(training_data)
	print("--- %s seconds for building tree---" % (time.time() - start_time))

	wrong_count = 0
	for row in testing_data:
		actual = row[-1]
		predicted = get_label(Sort_data(row, my_tree))
		# print ("predicted = ",predicted)
		# print ("Actual: %s. Predicted: %s" % (actual, predicted))
		if (actual != predicted):
			wrong_count += 1

	print ("wrong count = ",wrong_count)	



