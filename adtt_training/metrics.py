"""Pooled IoU with the project convention: zero unions contribute zero."""
import torch


class Confusion:
    def __init__(self, names, free_label):
        self.names, self.free = names, free_label
        self.matrix = torch.zeros((len(names), len(names)), dtype=torch.int64)

    def update(self, prediction, target, mask=None):
        prediction, target = prediction.reshape(-1).long(), target.reshape(-1).long()
        valid = target != 255
        if mask is not None:
            valid &= mask.reshape(-1).bool()
        prediction, target = prediction[valid], target[valid]
        count = len(self.names)
        if bool(((prediction < 0) | (prediction >= count) | (target < 0) | (target >= count)).any()):
            raise ValueError("Invalid metric label")
        self.matrix += torch.bincount(target * count + prediction,
                                     minlength=count * count).reshape(count, count).cpu()

    def result(self):
        matrix = self.matrix
        unions = matrix.sum(0) + matrix.sum(1) - matrix.diag()
        ious = [float(matrix[i, i] / unions[i]) if unions[i] else 0.0
                for i in range(len(self.names))]
        occupied = [i for i in range(len(self.names)) if i != self.free]
        tp = int(matrix[occupied][:, occupied].sum())
        fp = int(matrix[self.free, occupied].sum())
        fn = int(matrix[occupied, self.free].sum())
        values = [ious[i] for i in occupied]
        return dict(
            scale="0 to 1", voxels=int(matrix.sum()),
            geometry_IoU=tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            semantic_mIoU_nonfree=sum(values) / len(values),
            semantic_mIoU_including_free=sum(ious) / len(ious),
            nonfree_classes_with_union=sum(bool(unions[i]) for i in occupied),
            nonfree_classes_in_mean=len(occupied), per_class_IoU=dict(zip(self.names, ious)),
            per_class_union=dict(zip(self.names, unions.tolist())),
            absent_class_policy="Zero-union classes score zero and remain in the mean",
            confusion_GT_rows_prediction_columns=matrix.tolist())
