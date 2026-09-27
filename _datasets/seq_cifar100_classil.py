from _datasets import register_dataset
from _datasets.seq_cifar100 import SequentialCifar100


@register_dataset("seq-cifar100-classil")
class SequentialCifar100ClassIL(SequentialCifar100):
    """
    Class-Incremental Learning (Class-IL) variant of seq-cifar100.

    Identical client/task split as SequentialCifar100, but labels are kept
    in their GLOBAL range (0..99) instead of being remapped to the
    task-local range. Meant to be paired with a single-head ("shared
    classifier") network such as resnet18_classil, and evaluated WITHOUT an
    oracle task id at test time.
    """
    CLASS_IL = True


@register_dataset("joint-cifar100-classil")
class JointCifar100ClassIL(SequentialCifar100ClassIL):
    N_CLASSES_PER_TASK = 100
    N_TASKS = 1
