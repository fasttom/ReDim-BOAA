def get_unique_labels(data_loader):
    unique_labels = []
    for batch_idx, (data, target) in enumerate(data_loader):
        for label in target:
            if label not in unique_labels:
                unique_labels.append(label)
    return unique_labels