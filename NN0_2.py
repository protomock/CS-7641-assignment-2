"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
from __future__ import with_statement
import sys

sys.path.append(
    "/Users/chutchens/workspace/CS-7641-assignment-2/ABAGAIL/ABAGAIL.jar")

import os
import csv
import time

from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance, ConvergenceTrainer
from func.nn.backprop import RPROPUpdateRule, StochasticBackPropagationTrainer
from opt.example import NeuralNetworkOptimizationProblem

import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm

TRAIN_INPUT_FILE = os.path.join("m_trg.csv")
TEST_INPUT_FILE = os.path.join("m_trg.csv")
OUTFILE = "./NN_OUTPUT/"

INPUT_LAYER = 7
HIDDEN_LAYER = 4
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 1000


def initialize_instances(file):
    """Read the kickstarter.txt CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(file, "r") as kickstarter:
        reader = csv.reader(kickstarter)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(float(row[-1])))
            instances.append(instance)

    return instances


def train(oa, network, oaName, instances, measure):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print "\nError results for %s\n---------------------------" % (oaName,)
    i = 0
    while i < TRAINING_ITERATIONS:
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        i += oa.getIterations()
        print "%0.03f" % error
        print "%0.03f" % i


def main():
    """Run algorithms on the abalone dataset."""
    train_instances = initialize_instances(TRAIN_INPUT_FILE)
    test_instances = initialize_instances(TEST_INPUT_FILE)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(train_instances)
    rule = RPROPUpdateRule()

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["Backprop", "RHC", "SA", "GA"]
    results = ""

    for i in range(500):
        for name in oa_names:
            classification_network = factory.createClassificationNetwork(
                [INPUT_LAYER, HIDDEN_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
            networks.append(classification_network)
            nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

        oa.append(StochasticBackPropagationTrainer(
            data_set, classification_network, measure, rule))
        oa.append(RandomizedHillClimbing(nnop[1]))
        oa.append(SimulatedAnnealing(1E11, .95, nnop[2]))
        oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[3]))

        for i, name in enumerate(oa_names):
            start = time.time()
            correct = 0
            incorrect = 0

            trainer = ConvergenceTrainer(oa[i])
            train(trainer, networks[i], oa_names[i], train_instances, measure)
            end = time.time()
            training_time = end - start
            
            if name != "Backprop":
                optimal_instance = oa[i].getOptimal()
                networks[i].setWeights(optimal_instance.getData())

            start = time.time()
            for instance in test_instances:
                networks[i].setInputValues(instance.getData())
                networks[i].run()

                predicted = instance.getLabel().getContinuous()
                actual = networks[i].getOutputValues().get(0)

                if abs(predicted - actual) < 0.5:
                    correct += 1
                else:
                    incorrect += 1

            end = time.time()
            testing_time = end - start

            results += "\nResults for %s: \nCorrectly classified %d instances." % (
                name, correct)
            results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (
                incorrect, float(correct)/(correct+incorrect)*100.0)
            results += "\nTraining time: %0.03f seconds" % (training_time,)
            results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

            txt = "{},{},{}\n".format(
                float(correct)/(correct+incorrect)*100.0, training_time, testing_time)
            with open(OUTFILE + str(name), "a+") as f:
                f.write(txt)

        print results


if __name__ == "__main__":
    main()
