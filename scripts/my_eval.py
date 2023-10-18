import json

generation_file = ''


def eval_gen_only_1(generation_file):
    count = 0
    true_positive, pred_positive, pred_right = 0, 0, 0
    with open(generation_file, 'r') as f:
        for line in f.readlines():
            item = json.loads(line)
            output = item['output']
            output_ = [output]
            labels = item['all_outputs']
            if output in labels:
                count += 1
            true_positive += len(labels)
            pred_positive += 1
            for idx in labels:
                if idx in output_:
                    pred_right += 1

    p = pred_right / (pred_positive + 1e-08)
    r = pred_right / (true_positive + 1e-08)
    f = 2 * p * r / (p + r + 1e-08)
    return p, r, f, true_positive, pred_right, count


def eval_gen_morethan_1(generation_file):
    count = 0
    true_positive, pred_positive, pred_right = 0, 0, 0
    with open(generation_file, 'r') as f:
        for line in f.readlines():
            item = json.loads(line)
            output = item['output']
            if ',' in output:
                output = output.split(',')
            else:
                output = [output]

            labels = item['all_outputs']
            assert len(labels) == 1
            labels = labels[0]
            if ',' in labels:
                labels = labels.split(',')
            else:
                labels = [labels]

            true_positive += len(labels)
            pred_positive += len(output)

            for item in labels:
                if item in output:
                    pred_right += 1

            for item in output:
                if item in labels:
                    count += 1

    p = pred_right / (pred_positive + 1e-08)
    r = pred_right / (true_positive + 1e-08)
    f = 2 * p * r / (p + r + 1e-08)
    return p, r, f, true_positive, pred_right, count


def doc_test_eval(generation_file):
    count = 0
    true_positive, pred_positive, pred_right = 0, 0, 0
    with open(generation_file, 'r') as f:
        for line in f.readlines():
            item = json.loads(line)
            output = item['output']
            if ',' in output:
                output = output.split(',')
            else:
                output = [output]
            labels = item['all_outputs']
            true_positive += len(labels)
            pred_positive += len(output)

            for item in labels:
                if item in output:
                    pred_right += 1

            for item in output:
                if item in labels:
                    count += 1

    p = pred_right / (pred_positive + 1e-08)
    r = pred_right / (true_positive + 1e-08)
    f = 2 * p * r / (p + r + 1e-08)
    return p, r, f, true_positive, pred_right, count


def eval_gen_morethan_1_v2(generation_file):
    count = 0
    true_positive, pred_positive, pred_right = 0, 0, 0
    with open(generation_file, 'r') as f:
        for line in f.readlines():
            item = json.loads(line)
            dia_length = len(item['metadata']['context'])
            output = item['output']
            if ',' in output:
                output = output.split(',')
            else:
                output = [output]

            for s in output:
                if not s.isdigit() and str(dia_length) not in output:
                    output.append(str(dia_length))

            labels = item['all_outputs']
            assert len(labels) == 1
            labels = labels[0]
            if ',' in labels:
                labels = labels.split(',')
            else:
                labels = [labels]

            true_positive += len(labels)
            pred_positive += len(output)

            for item in labels:
                if item in output:
                    pred_right += 1

            for item in output:
                if item in labels:
                    count += 1

    p = pred_right / (pred_positive + 1e-08)
    r = pred_right / (true_positive + 1e-08)
    f = 2 * p * r / (p + r + 1e-08)
    return p, r, f, true_positive, pred_right, count


p, r, f, true_positive, pred_right, count = eval_gen_morethan_1_v2(generation_file)
print(p, r, f, true_positive)
print(pred_right, count)
