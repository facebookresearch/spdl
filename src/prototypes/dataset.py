import logging
import time

from spdl.dataset import _dataset
from spdl.dataset.imagenet import ImageNet
from spdl.dataset.librispeech import LibriSpeech


logging.basicConfig(level=logging.DEBUG)


def _test(dataset):
    print(len(dataset))
    for i, item in enumerate(dataset):
        print(item)
        assert item == dataset[item._index]
        if i > 10:
            print(" breaking")
            break

    def _test_slice(s):
        print(s)
        for i, item in enumerate(dataset[s]):
            print(item)

            assert item == dataset[item._index]
            if i > 10:
                print(" breaking")
                break

    _test_slice(slice(0, 10, 2))
    _test_slice(slice(0, 10))
    _test_slice(slice(1281165, None, None))
    _test_slice(slice(None, 10))
    _test_slice(slice(None, 25, 3))
    _test_slice(slice(6, 31, 7))
    _test_slice(slice(None, None, None))


# dataset = ImageNet("imagenet.db", split="train")
dataset = LibriSpeech("librispeech.db", split="test-other")

print(dataset)
t0 = time.monotonic()
dataset.shuffle()
elapsed = time.monotonic() - t0
print(dataset)
print(f"{elapsed} [sec]")


for col in dataset.attributes:
    if col == "sample_rate":
        continue
    for desc in [True, False]:
        print(f"sort by {col}, {desc=}")
        t0 = time.monotonic()
        dataset.sort(col, desc=desc)
        elapsed = time.monotonic() - t0
        print(dataset)
        print(f"{elapsed} [sec]")
print(dataset.attributes)

print(len(dataset))
# _test(dataset)
datasets = _dataset.split(dataset, 3, "librispeech_split_{}.db")
for ds in datasets:
    print(ds.attributes)
    print(ds)
# dataset = ImageNet(split="train", path="imagenet.db")
# dataset.shuffle()
# print(dataset)

for i, batch in enumerate(dataset.iterate(batch_size=2, max_batch=5)):
    print(i, f"{len(batch)}")
    print("\n".join(f"{item}, {item._index}" for item in batch))
    print()
