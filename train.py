import argparse
import sys

# Decision tree files
from decisionTreeTrain import *

# Adaboost
from adaBoostTrain import *


examples = ""
hypothesisOut = ""
learningType = ""


def main():
    '''
    Main function for testing the program
    :return:
    '''
    if len(sys.argv) <= 3:
        print('Usage: {} <examples> <hypothesisOut> <learning-type>'.format(
            sys.argv[0]))
        exit(1)

    examples = sys.argv[1]
    hypothesisOut = sys.argv[2]
    learningType = sys.argv[3]

    # Train using Decision Tree
    if learningType == "dt":
        decisionTreeFit(examples=examples, hypothesisOut=hypothesisOut)
    elif learningType == "ada":
        ada_complete_predict(examples=examples, hypothesisOut=hypothesisOut)
        #hypot_weights, stump_val = adaBoostTrain_main(num_iter=8)
        #print(ada_predict("processed_data/binary_dutch_eng_validation.csv", hypot_weights, stump_val))

    print(examples)
    print(hypothesisOut)
    print(learningType)


main()