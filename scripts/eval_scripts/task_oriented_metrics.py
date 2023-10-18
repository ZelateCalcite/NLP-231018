def joint_accuracy(data):
    joint_acc = 0
    for dp in data:
        prediction = str(dp['output']).lower()
        label = dp['all_outputs'][0]
        prediction = sorted([x.strip() for x in prediction.split(',')])
        label = sorted([x.strip() for x in label.split(',')])

        print('predicted')
        print(prediction)
        print('actual')
        print(label)
        print()

        if set(prediction) == set(label):
            joint_acc += 1

    joint_acc /= len(data)
    return joint_acc
