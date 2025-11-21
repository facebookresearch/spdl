import unittest

from spdl_unittest.cuda.buffer_transfer_test import (
    TestArrayTransfer,
    TestTransferBufferToCuda,
    TestTransferCpu,
)
from spdl_unittest.cuda.nvdec_video_decoding_test import (
    TestNvdecBasic,
    TestNvdecH264,
)


def main():
    suite = unittest.TestSuite()
    cases = [
        # TestTransferBufferToCuda("test_transfer_buffer_to_cuda_0_audio"),
        # TestTransferBufferToCuda("test_transfer_buffer_to_cuda_1_video"),
        # TestTransferBufferToCuda("test_transfer_buffer_to_cuda_2_image"),
        # TestTransferBufferToCuda(
        #     "test_transfer_buffer_to_cuda_with_pytorch_allocator_0_audio"
        # ),
        # TestTransferBufferToCuda(
        #     "test_transfer_buffer_to_cuda_with_pytorch_allocator_1_video"
        # ),
        # TestTransferBufferToCuda(
        #     "test_transfer_buffer_to_cuda_with_pytorch_allocator_2_image"
        # ),
        # TestArrayTransfer("test_array_transfer_non_contiguous_numpy"),
        # TestArrayTransfer("test_array_transfer_non_contiguous_torch"),
        # TestArrayTransfer("test_array_transfer_numpy"),
        # TestArrayTransfer("test_array_transfer_smoke_test"),
        # TestArrayTransfer("test_array_transfer_torch"),
        # TestTransferCpu("test_transfer_cpu"),
        # TestNvdecBasic("test_nvdec_negative"),
        # TestNvdecBasic("test_nvdec_no_file"),
        # TestNvdecBasic("test_nvdec_odd_size"),
        # TestNvdecBasic("test_nvdec_video_smoke_test"),
        TestNvdecH264("test_nvdec_decode_crop_resize"),
        TestNvdecH264("test_nvdec_decode_h264_420p_basic"),
        TestNvdecH264("test_nvdec_decode_h264_420p_crop"),
    ]

    suite.addTests(cases)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    if result.failures:
        print("\n--- Failures ---")
        for test, traceback_info in result.failures:
            print(f"Test: {test}")
            print(f"Traceback:\n{traceback_info}")

    if result.errors:
        print("\n--- Errors ---")
        for test, traceback_info in result.errors:
            print(f"Test: {test}")
            print(f"Traceback:\n{traceback_info}")

    if result.failures or result.errors:
        print(
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
            "@@@ Test failed\n"
            "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
        )
        exit(1)


if __name__ == "__main__":
    main()
