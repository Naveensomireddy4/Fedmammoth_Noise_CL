from _datasets import register_dataset
from _datasets.seq_tinyimagenet import SequentialTinyImageNet


@register_dataset("seq-tinyimagenet-classil")
class SequentialTinyImageNetClassIL(SequentialTinyImageNet):
    """
    Class-Incremental Learning (Class-IL) variant of seq-tinyimagenet.

    Identical client/task split as SequentialTinyImageNet, but labels are
    kept in their GLOBAL range (0..199) instead of being remapped to the
    task-local range. Meant to be paired with a single-head ("shared
    classifier") network such as resnet18_classil, and evaluated WITHOUT an
    oracle task id at test time.
    """
    CLASS_IL = True


@register_dataset("joint-tinyimagenet-classil")
class JointTinyImageNetClassIL(SequentialTinyImageNetClassIL):
    N_CLASSES_PER_TASK = 200
    N_TASKS = 1
