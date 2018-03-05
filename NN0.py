"""
Backprop NN training on Madelon data (Feature selection complete)

"""
import os
import csv
import time
import sys
sys.path.append(
    "/Users/chutchens/workspace/CS-7641-assignment-2/ABAGAIL/ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, StochasticBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import LogisticSigmoid

# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 7
HIDDEN_LAYER = 4
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 100
OUTFILE = './NN_OUTPUT/BACKPROP_LOG.txt'


def initialize_instances(infile):
    """Read the m_trg.csv CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(float(row[-1])))
            instances.append(instance)

    return instances


def errorOnDataSet(network, ds, measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()

        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted, 1), 0)
        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    print "\nCorrect %s\n---------------------------" % (correct,)
    print "\nIncorrect %s\n---------------------------" % (incorrect,)
    print "\nN %s\n---------------------------" % (N,)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)
    return MSE, acc


def train(oa, network, oaName, training, validation, test, measure):
    """Train a given network on a set of instances.
    """
    print "\nError results for %s\n---------------------------" % (oaName,)
    for iteration in xrange(TRAINING_ITERATIONS):
        oa.train()

        error = 0.00
        for instance in training[0:40000]:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        print("%0.03f" % error)
        if iteration % 10 == 0:
            # MSE_trg, acc_trg = errorOnDataSet(network, training[0:40000], measure)
            MSE_val, acc_val = errorOnDataSet(network, validation, measure)
    #         MSE_tst, acc_tst = errorOnDataSet(network, testing_ints, measure)
    #         txt = '{},{},{},{},{},{},{},{}\n'.format(iteration, MSE_trg, MSE_val, MSE_tst, acc_trg, acc_val, acc_tst, times[-1])
            print MSE_val, acc_val
    #         with open(OUTFILE, 'a+') as f:
    #             f.write(txt)
     
    # times = [0]
    # for iteration in xrange(TRAINING_ITERATIONS):
    #     start = time.clock()
    #     oa.train()
    #     elapsed = time.clock()-start
    #     times.append(times[-1]+elapsed)



def main():
    """Run this experiment"""
    training_ints = initialize_instances('m_trg.csv')
    testing_ints = initialize_instances('m_test.csv')
    validation_ints = initialize_instances('m_val.csv')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    sigmoid = LogisticSigmoid()
    rule = RPROPUpdateRule()
    oa_names = ["Backprop"]
    classification_network = factory.createClassificationNetwork([INPUT_LAYER,
                                                                  HIDDEN_LAYER,
                                                                  OUTPUT_LAYER], sigmoid)
    trainer = StochasticBackPropagationTrainer(data_set, classification_network, measure, rule)                                                                  
    train(trainer,
          classification_network, 'Backprop', training_ints, validation_ints, testing_ints, measure)


if __name__ == "__main__":
    with open(OUTFILE, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg',
                                                   'MSE_val', 'MSE_tst', 'acc_trg', 'acc_val', 'acc_tst', 'elapsed'))
    main()
