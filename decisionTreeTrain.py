"""
Author: Jennifer Liu
File: Lab 2 Decision Tree
"""

# Library to do simple arithmetic operations, and do
# rounding (ceil, floor)
import math

# Library for loading a csv file and converting
# it into a dictionary
import pandas as pd
import numpy as np

# Custom functions from features
from features import *

"""
Load the binary dataset
and save it as a global variable
so it is accessible for the entire program
"""
survey_binary = None

# The variable we are trying to predict
target = None

# Create a gloabl point for the classifier program that we
# will be writing
output_file_handle = None

# The depth of tree
DEPTH = 1

# When doing adaboosting, we should
# consider giving weights to different
# trees
WEIGHT = None


def entropy(count1, count2):
    """
    Calculate the entropy of the two counts

    Param:
        Count1 - # of counts in set 1
        Count2 - # of counts in set 2
    """

    entropy_v = 0.0

    if count1 != 0 or count2 != 0:
        # The total number of values
        total = count1 + count2

        # Log(0) returns an error, so this code handles Log(0) errors

        # If both counts == 0, then the result is extremely impure
        # so I give it a 1
        if count1 == 0 and count2 == 0:
            entropy_v = 1.0
        # If one of the counts = 0, I consider the count[0] = 0
        elif count1 == 0:
            entropy_v = -(0 + (count2 / total) * math.log2((count2 / total)))
        # If one of the counts = 1, I consider the count[1] = 1
        elif count2 == 0:
            entropy_v = -((count1 / total) * math.log2((count1 / total)) + 0)
        # None of the counts are 0
        else:
            entropy_v = -(
                        (count1 / total) * math.log2((count1 / total)) + (count2 / total) * math.log2((count2 / total)))

    # Using the entropy_v above, if it is a 0, it represents as a -0.0
    # So I had an exception to return 0.0 if entropy_v == -0.0
    if entropy_v == -0.0:
        return 0.0
    else:
        return entropy_v


def find_best_split(record_datas, target):
    """
    Find the best question to ask by iterating over every attribute and
    calculating the minimum information gain
    """
    attributes = record_datas.keys()
    best_attribute = attributes[0]  # Keep track of the best attribute that produce the best entropy
    n_attributes = record_datas.shape[1]
    min_info_gain = float("inf")  # Storing the min entropy
    prev_info_gain = 0
    best_split = None
    if_struc = None


    for col in range(n_attributes):  # For each attribute

        # If attribute being inspected is Lang
        # we do not inspect
        if attributes[col] == target:
            continue

        # Get the name of attribute
        attribute = attributes[col]

        attribute_true = record_datas[attribute] == 1  # records where attribute == True
        target_true = record_datas[target] == 1  # Records where target variable == True
        attribute_false = record_datas[attribute] == 0  # Records where attribute == False
        target_false = record_datas[target] == 0  # Records where target variable == False

        tp_rows = record_datas[attribute_true & target_true]  # Attribute == True, Target == True
        tn_rows = record_datas[attribute_false & target_false]  # Attribute == False, Target == False
        fp_rows = record_datas[attribute_true & target_false]  # Attribute == True, Target == False
        fn_rows = record_datas[attribute_false & target_true]  # Attribute == False, Target == True

        fn_count = fn_rows.shape[0]  # False negative count
        fp_count = fp_rows.shape[0]  # False positive count
        tp_count = tp_rows.shape[0]  # True positive count
        tn_count = tn_rows.shape[0]  # True negative count

        # Find the max bewtween TP and FP counts
        true_max = max(tp_count, fp_count)

        # Find the best between FN and TN Count
        false_max = max(fn_count, tn_count)

        # If true max is greater than false max
        # we see if the TP >= FP
        # My Default is when Attribute == True, Target == True
        if true_max >= false_max:
            # if attribute == True, target == True
            if tp_count >= fp_count:
                if_struc = "true_true"
            # if attribute == True, target == False
            else:
                if_struc = "true_false"
        else:
            # if attribute == False, target == True
            if fn_count >= tn_count:
                if_struc = "false_true"
            # if attribute == False, target == False
            else:
                if_struc = "false_false"

        # If the weight is specified
        if WEIGHT is not None:
            # Debugger to know that I cam here
            #print("Weighted sum is activated")

            fn_count = 0  # False negative count
            fp_count = 0  # False positive count
            tp_count = 0  # True positive count
            tn_count = 0  # True negative count


            # Calculating weighted entropy
            # Traverse the record and find
            for idx in range(record_datas.shape[0]):
                # True positive
                if record_datas.iloc[idx][attribute] == 1 and record_datas.iloc[idx][target] == 1:
                    # Calculate W*P(x|tp)*log2(P(x|tp))
                    tp_count += WEIGHT[idx]
                # TN
                elif record_datas.iloc[idx][attribute] == 0 and record_datas.iloc[idx][target] == 0:
                    # Calculate W*P(x|tn)*log2(P(x|tn))
                    #weighted_entropy_arr.append(-(1 / tn_count) * math.log((1 / tn_count), 2))
                    tn_count += WEIGHT[idx]
                # FP
                elif record_datas.iloc[idx][attribute] == 1 and record_datas.iloc[idx][target] == 0:
                    # Calculate W*P(x|tn)*log2(P(x|tn))
                    #weighted_entropy_arr.append(-(1 / fp_count) * math.log((1 / fp_count), 2))
                    fp_count += WEIGHT[idx]
                # FN
                elif record_datas.iloc[idx][attribute] == 0 and record_datas.iloc[idx][target] == 1:
                    # Calculate W*P(x|tn)*log2(P(x|tn))
                    #weighted_entropy_arr.append(-(1 / fn_count) * math.log((1 / fn_count), 2))
                    fn_count += WEIGHT[idx]

                # Find the max bewtween TP and FP counts
                true_max = max(tp_count, fp_count)

                # Find the best between FN and TN Count
                false_max = max(fn_count, tn_count)

                # If true max is greater than false max
                # we see if the TP >= FP
                # My Default is when Attribute == True, Target == True
                if true_max >= false_max:
                    # if attribute == True, target == True
                    if tp_count >= fp_count:
                        if_struc = "true_true"
                    # if attribute == True, target == False
                    else:
                        if_struc = "true_false"
                else:
                    # if attribute == False, target == True
                    if fn_count >= tn_count:
                        if_struc = "false_true"
                    # if attribute == False, target == False
                    else:
                        if_struc = "false_false"

        #print("No weights activated")

        # If current entropy is smaller than min,
        # update the entropy, and best attribute
        # also note the best split point
        false_entropy = entropy(fn_rows.shape[0], tn_rows.shape[0])
        true_entropy = entropy(tp_rows.shape[0], fp_rows.shape[0])

        total = fn_count + fp_count + tp_count + tn_count  # Total number of records

        # If there are no data, skip this attribute
        if total == 0:
            continue

        # Mixed entropy from left and right node
        new_min_info = ((fn_count + tn_count) / total) * false_entropy + ((fp_count + tp_count) / total) * true_entropy

        # If current entropy is smaller than min,
        # update the entropy, and best attribute
        # also note the best split point
        if new_min_info < min_info_gain:
            min_info_gain, best_attribute = new_min_info, attribute
            best_split = (partition(record_datas, attribute))

    #print("Final Smallest Min Entropy")
    #print(min_info_gain)
    #print("Final Best Attribute")
    #print(best_attribute)
    #print()

    # return all sorts of important values
    return min_info_gain, best_attribute, best_split, if_struc


def partition(record_datas, attribute):
    """
    Partition the dataset

    For each row in the dataset, check if it matches the attribute. If so,
    add it to the 'true rows', otherwise, add it to 'false rows'
    """

    # if the record for that attribute is 1, it is true
    # otherwise it is false
    true_rows = record_datas[record_datas[attribute] == 1]
    false_rows = record_datas[record_datas[attribute] == 0]

    return true_rows, false_rows


def remove_nonbinary_col(record_datas, target):
    """
    There were datas in the the file that
    were non-binary. Therefore, I wrote a function
    to remove those columns
    """
    modified_records = record_datas

    # Traverse through every attribute,
    # if the attribute has non binary data,
    # drop the column from record_datas
    for attribute in record_datas.keys():
        if record_datas[attribute].iloc[0] != 0 and record_datas[attribute].iloc[0] != 1:
            modified_records = modified_records.drop(attribute, axis=1)

    return modified_records


def class_counts(record_datas):
    """
    For the attribute provided in the parameter
    Separate the data into records that have a 1 for that attribute
    and records that have a 0 for that attribute

    Return a dictionary representation of that record
    """
    return {1: record_datas[record_datas == 1].shape[0], 0: record_datas[record_datas == 0].shape[0]}

class Leaf:
    """
    Holds a dictionary of class (eg. "vowelDutch") -> # of times
    it appears in the rows from the training data that reach this leaf.
    """
    def __init__(self, record_data, attribute, target_name, is_left, if_struc):
        self.record_data = record_data
        self.attribute = attribute
        self.target_name = target_name
        self.is_left = is_left
        self.if_struc = if_struc
        self.predictions = {1: len(partition(record_data, "Lang")[0]), 0: len(partition(record_data, "Lang")[1])}

class Decision_Node:
    """
    A Decision node asks a question. It holds a reference to the
    question, and to the two child nodes
    """

    def __init__(self, attribute, true_branch, false_branch, depth):
        self.attribute = attribute
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth


def build_tree(record_datas, target_name, depth, is_left, if_struc_p):
    """
    Build the decision tree using recursion
    Param:
        record_datas: The data
        target_name: the name of the target variable
        depth: What tree depth we are at so we can print data properly
    """

    min_info_gain, attribute, best_split, if_struc = find_best_split(record_datas, target_name)
    # print("-------------------------------------------------------")

    # If the depth of this tree reaches Depth d, stop building the tree
    # or if there is nothing else to split on
    if depth == DEPTH or best_split is None:
        if if_struc_p == "":
            return Leaf(record_datas, attribute, target_name, is_left, if_struc)
        else:
            return Leaf(record_datas, attribute, target_name, is_left, if_struc_p)

    best_right = best_split[0]  # Best True branch at this instance
    best_left = best_split[1]  # Best False Branch at this instance

    # Continuously build the right/true branch
    true_branch = build_tree(best_right, target_name, depth + 1, False, if_struc)

    # Continously build the left/false branch
    false_branch = build_tree(best_left, target_name, depth + 1, True, if_struc)

    # Create a Decision Node structure for every branch
    return Decision_Node(attribute, true_branch, false_branch, depth)


def print_tree(node, output_file_handle, spacing=""):
    """Print the tree structure to a file"""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        # print (spacing + "Predict", node.predictions)
        # output_file_handle.write(spacing + node.if_struc + "\n")
        # If the number of english is greater than the number of
        # dutch, it is english
        if node.predictions[0] >= node.predictions[1]:
            output_file_handle.write(spacing + target + " = 0\n")
        else:
            output_file_handle.write(spacing + target + " = 1\n")
        return

    # Print the question at this node
    output_file_handle.write(spacing + "if datas.iloc[data_record]['" + str(node.attribute) + "'] <= 0:\n")

    # Call this function recursively on the false branch
    # print (spacing + '--> False:')
    print_tree(node.false_branch, output_file_handle, spacing + "  ")

    output_file_handle.write(spacing + "else:\n")
    # Call this function recursively on the true branch
    # print (spacing + '--> True:')
    print_tree(node.true_branch, output_file_handle, spacing + "  ")


def dt_predict_helper(node, data_record):
    """
    Main function for predicting
    :param node:
    :return:
    """
    if isinstance(node, Leaf):
        if node.is_left:
            if node.if_struc == "true_true" or node.if_struc == "false_false":
                return 0
            elif node.if_struc == "true_false" or node.if_struc == "false_true":
                return 1
        else:
            if node.if_struc == "true_true" or node.if_struc == "false_false":
                return 1
            elif node.if_struc == "true_false" or node.if_struc == "false_true":
                return 0
        return

    if data_record[node.attribute] <= 0:
        return dt_predict_helper(node.false_branch, data_record)
    else:
        return dt_predict_helper(node.true_branch, data_record)


def dt_predict(node, datas, single=False):
    """
    Main function to predict whether
    it is eng or dutch given variables
    :param node:
    :param datas:
    :return:
    """
    predictions = []

    if single:
        return dt_predict_helper(node, datas)
    else:
        for data_record in range(datas.shape[0]):
            predictions.append(dt_predict_helper(node, datas.iloc[data_record]))

        return np.array(predictions)

def print_tree_to_output(node, spacing=""):
    """Print the tree to console"""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        # print (spacing + "Predict", node.predictions)
        # output_file_handle.write(spacing + node.if_struc + "\n")
        # If the number of english is greater than the number of
        # dutch, it is english
        return node.predictions

    # Print the question at this node
    print(spacing + "if datas.iloc[data_record]['" + str(node.attribute) + "'] <= 0:")

    # Call this function recursively on the false branch
    # print (spacing + '--> False:')
    print_tree_to_output(node.false_branch, spacing + "  ")

    print(spacing + "else:")
    # Call this function recursively on the true branch
    # print (spacing + '--> True:')
    print_tree_to_output(node.true_branch, spacing + "  ")


def emit_header():
    """
    Open up a file pointer to a file named HW_NN_LastName_FirstName_Classifier
    """
    # Write a line at the end of the file
    #output_file_handle.seek(0, 2)

    # Write a program that writes the program
    output_file_handle.write("# Library for loading a csv file and converting\n" \
                             "# it into a dictionary\n" \
                             "import pandas as pd\n" \
                             "import csv\n" \
                             "def decisionTree_predict(csv_file):\n" \
                             "\tdatas = pd.read_csv(csv_file)\n" \
                             "\tTP = 0\n" \
                             "\tTN = 0\n" \
                             "\twith open('Lab_02_Liu_Jennifer_ResultClassifications.csv', 'w') as result_csv:\n" \
                             "\t\twriter = csv.writer(result_csv)\n" \
                             "\t\twriter.writerow(['result'])\n" \
                             "\t\tfor data_record in range(datas.shape[0]):\n")


def emit_classifier_call():
    """
    This will create the main function in the classifier
    It creates the classifier program
    It writes the result to the CSV
    It calculates the prediction accuracy on Validation data
    """

    # Build tree
    decision_tree = build_tree(survey_binary, target, 0, False, "")

    # print the resulting tree to the classifier program
    print_tree(decision_tree, output_file_handle, "\t\t\t")

    # Converting binary numbers into nl and en values
    convert = "\t\t\tresult = 'en' if " + target + " == 1 else 'nl'\n"

    # Write the converter to file
    output_file_handle.write(convert)

    # Print prediction output
    # If Classifier thinks it is English, print 1
    # else print 0
    output_file_handle.write("\t\t\tprint(result)\n")

    # Write the result to csv file
    output_file_handle.write("\t\t\twriter.writerow([" + target + "])\n")

    # output_file_handle.write("\t\tresult_csv.write(str(" + target + ") + '\n')\n")

    # Get statistics value for our classifier
    # Calculate the TP and TN
    output_file_handle.write("\t\t\tif " + target + " == datas.iloc[data_record]['" + target + "']:\n" \
                                                                                               "\t\t\t\tif " + target + " == 0:\n" \
                                                                                                                        "\t\t\t\t\tTN += 1\n" \
                                                                                                                        "\t\t\t\telse:\n" \
                                                                                                                        "\t\t\t\t\tTP += 1\n" \
                                                                                                                        "\n" \
                                                                                                                        "\t\tresult_csv.close()\n"
                                                                                                                        "\t\tprint('TP: ' + str(TP) + ' TN: ' + str(TN))\n" \
                                                                                                                        "\t\taccuracy = (TP + TN)/datas.shape[0]\n" \
                                                                                                                        "\t\tprint('Accuracy: ' + str(accuracy))\n")

    return decision_tree

def emit_trailer(csv_file):
    output_file_handle.write("\ndecisionTree_predict(\"" + csv_file + "\")")


def decisionTreeFit(examples="processed_data/train.txt", hypothesisOut="Classifier.py", depth=2, weight=None):

    # Convert example file into a csv file
    write_to_csv("processed_data/binary_dutch_eng.csv", examples)

    # Load Data
    global survey_binary
    survey_binary = pd.read_csv('processed_data/binary_dutch_eng.csv')

    # Set Target Variable
    global target
    target = "Lang"

    # Specify depth
    global DEPTH
    DEPTH = depth

    # Specify weight for adaboosting
    global WEIGHT
    WEIGHT = weight

    # open a file to write the classifier to
    global output_file_handle
    output_file_handle = open(hypothesisOut, "w")
    # output_file_handle = open("HW_06_Liu_Jennifer_Classifier_Test.py", "w+")

    # Emit header = Include import functions
    emit_header()

    # Create the classifier
    decision_tree = emit_classifier_call()

    # Call the classifier function and get validation accuracy
    emit_trailer("processed_data/binary_dutch_eng_validation.csv")

    # Close Classifier file
    output_file_handle.close()

    return decision_tree

decisionTreeFit(depth=2)