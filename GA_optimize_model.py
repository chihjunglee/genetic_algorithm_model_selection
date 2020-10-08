from pyeasyga import pyeasyga
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix
import random

# with open('model_evaluation/pathogenicity_y_true.pickle', 'rb') as file:
#     y_true = pickle.load(file)
#
# print('y_true:', len(y_true))

with open('model_evaluation/pathogenicity_AUC_model_pred_dict.pickle', 'rb') as file:
    pathogenic_model_pred_dict = pickle.load(file)

y_true = pathogenic_model_pred_dict['SIFT']['y_true']

# setup data
data = []
for model in list(pathogenic_model_pred_dict.keys()):
    # print(model)
    item = {'model': model,
            'y_pred': pathogenic_model_pred_dict[model]['y_pred']}
    data.append(item)
# print(data)


fp = open('GA_process_3.txt', 'w')


def fitness(individual, data):
    print('-----')
    selected_model_count = 0
    selected_model_pred_list = []
    value = 0
    print(individual)

    for selected, model in zip(individual, data):
        # print('-----')

        if selected:
            selected_model_count += 1
            # print(model.get('model'))
            # print(len(model.get('y_pred')))
            selected_model_pred_list.append(model.get('y_pred'))

    if len(selected_model_pred_list) == 0:  # if no model be selected [all 0]
        value = 0
        return value

    else:
        print(len(selected_model_pred_list))

        valid_model = []
        voting_list = []
        for pred_idx in range(len(selected_model_pred_list[0])):
            count_valid = 0
            vote = 0

            for i in range(len(selected_model_pred_list)):
                pred = selected_model_pred_list[i][pred_idx]
                if pred != '.':
                    count_valid += 1
                    vote += pred

            valid_model.append(count_valid)
            if count_valid != 0:
                if vote / count_valid > 0.5:  # More than half of the votes
                    voting_list.append(1)
                else:
                    voting_list.append(0)
            else:
                voting_list.append('.')

        y_pred = []
        tmp_y_true = []

        for i in range(len(voting_list)):
            pred = voting_list[i]
            if pred != '.':
                y_pred.append(pred)
                tmp_y_true.append(y_true[i])

        auc_score = roc_auc_score(tmp_y_true, y_pred)
        print(auc_score)

        value = auc_score

        fp.write(str(len(selected_model_pred_list)) + '\n')
        fp.write('AUC:' + str(value) + '\n')
        fp.write(str(selected_model_pred_list))
        fp.write('\n')

        return value


ga = pyeasyga.GeneticAlgorithm(data, population_size=100, generations=100)  # define GA object initial parameters


# ga = pyeasyga.GeneticAlgorithm(data, population_size=5, generations=10)
# ga = pyeasyga.GeneticAlgorithm(data,
#                                population_size=200,
#                                generations=100,
#                                crossover_probability=0.8,
#                                mutation_probability=0.2,
#                                elitism=True,
#                                maximise_fitness=False)

# def create_individual(data):
#     # initial first individual
#     return [random.randint(0, 1) for _ in range(len(data))]
#    # return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# ga.create_individual = create_individual

# set the GA's fitness function
ga.fitness_function = fitness
ga.run()
print(ga.best_individual())

# for individual in ga.last_generation():
#     print(individual)

for idx in range(len(ga.best_individual()[1])):
    if ga.best_individual()[1][idx] == 1:
        print(list(pathogenic_model_pred_dict.keys())[idx])

with open('GA_optimal_result/GA_last_generation_new.txt', 'w') as f:
    for individual in ga.last_generation():
        print(type(individual))
        print(individual)

        f.write('AUC:' + str(individual[0]) + '\n')
        f.write(str(individual[1]))
        f.write('\n')

        model_list = []
        for idx in range(len(individual[1])):
            if individual[1][idx] == 1:
                model_list.append(list(pathogenic_model_pred_dict.keys())[idx])
                # print(list(pathogenic_model_pred_dict.keys())[idx])

        f.write(str(model_list))
        f.write('\n')
