from _datasets import register_dataset
from _datasets.seq_cifar10 import SequentialCifar10


@register_dataset("seq-cifar10-classil")
class SequentialCifar10ClassIL(SequentialCifar10):
    """
    Class-Incremental Learning (Class-IL) variant of seq-cifar10.

    Identical client/task split as SequentialCifar10, but labels are kept in
    their GLOBAL range (0..N_CLASSES-1) instead of being remapped to the
    task-local range. Meant to be paired with a single-head ("shared
    classifier") network such as resnet18_classil, and evaluated WITHOUT an
    oracle task id at test time.
    """
    CLASS_IL = True


@register_dataset("joint-cifar10-classil")
class JointCifar10ClassIL(SequentialCifar10ClassIL):
    N_CLASSES_PER_TASK = 10
    N_TASKS = 1
