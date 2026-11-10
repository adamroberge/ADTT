"""Pooled all-voxel or explicitly masked IoU; absent unions remain undefined."""
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
        ious = [float(matrix[i, i] / unions[i]) if unions[i] else None
                for i in range(len(self.names))]
        occupied = [i for i in range(len(self.names)) if i != self.free]
        tp = int(matrix[occupied][:, occupied].sum())
        fp = int(matrix[self.free, occupied].sum())
        fn = int(matrix[occupied, self.free].sum())
        values = [ious[i] for i in occupied if ious[i] is not None]
        all_values = [value for value in ious if value is not None]
        return dict(
            scale="0 to 1", voxels=int(matrix.sum()),
            geometry_IoU=tp / (tp + fp + fn) if tp + fp + fn else None,
            semantic_mIoU_nonfree=sum(values) / len(values) if values else None,
            semantic_mIoU_including_free=sum(all_values) / len(all_values) if all_values else None,
            nonfree_classes_with_union=len(values), per_class_IoU=dict(zip(self.names, ious)),
            absent_class_policy="Exclude only zero-union classes; false-positive-only classes score zero",
            confusion_GT_rows_prediction_columns=matrix.tolist())
