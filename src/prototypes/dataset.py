import time
import logging

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



# dataset = ImageNet(split="train", path="imagenet.db")
dataset = LibriSpeech(split="test-other", path="librispeech.db")

print(dataset)
t0 = time.monotonic()
dataset.shuffle()
elapsed = time.monotonic() - t0
print(dataset)
print(f"{elapsed} [sec]")


for col in dataset.attributes:
    for desc in [True, False]:
        print(f"sort by {col}, {desc=}")
        t0 = time.monotonic()
        dataset.sort(col, desc=desc)
        elapsed = time.monotonic() - t0
        print(dataset)
        print(f"{elapsed} [sec]")
print(dataset.attributes)

print(len(dataset))
_test(dataset)
dataset = ImageNet(split="train", path="imagenet.db")
# dataset.shuffle()
print(dataset)

