import numpy as np


def calculate_fisher_discriminant(x_list, y_class, percentage_cutoff=0.75):
    assert 1.0 > percentage_cutoff >= 0.0
    count = {}
    x_dic = {}
    n_dimension = x_list.shape[1]
    for index in range(x_list.shape[0]):
        y = y_class[index]
        if y in count.keys():
            count[y] += 1
            x_dic[y].append(x_list[index])
        else:
            count[y] = 1
            x_dic[y] = [x_list[index]]

    means = {}
    sw = np.zeros((n_dimension, n_dimension))
    for y in count.keys():
        x_dic[y] = np.stack(x_dic[y], axis=1)
        means[y] = np.mean(x_dic[y], axis=1)
        sn = np.zeros((n_dimension, n_dimension))
        for sub_matrix in np.transpose(x_dic[y]):
            x_temp = sub_matrix - means[y]
            sn += np.matmul(x_temp, np.transpose(x_temp))
        sw += sn
    sw = np.linalg.pinv(sw)
    vs = {}
    samples = {}
    for y in count.keys():
        mean_sum = np.zeros((n_dimension))
        for y2 in count.keys():
            if y != y2:
                mean_sum += means[y2]
        mean_sum /= (len(means) - 1)
        vs[y] = np.matmul(sw, means[y] - mean_sum)
        sample = np.abs(np.matmul(vs[y], x_dic[y]))
        mn = np.min(sample)
        sample = (sample - mn) / (np.max(sample) + mn)
        samples[y] = sample
    output_x = []
    output_y = []
    for y in samples.keys():
        for index, subsample in enumerate(samples[y]):
            if subsample >= percentage_cutoff:
                output_x.append(x_dic[y][:, index])
                output_y.append(y)
    output_x = np.stack(output_x)
    output_y = np.stack(output_y)
    return output_x, output_y
