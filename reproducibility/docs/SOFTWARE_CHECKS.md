# Software checks

The tests exercise exact-K selection and ties, directional alignment/aggregation, N/I state sharing, FP64-to-FP32 order, foreground order, state and prediction validation, interruption/resume behavior and evaluation interfaces. Generated arrays, synthetic RGB/masks and API doubles isolate these operations. They do not load a pretrained model or estimate localization performance on a new dataset.

From the package root, use new external scratch directories:

```text
python -I -B code/run_reader_tests.py --scratch ../diream-tests/core --suite core
python -I -B code/run_reader_tests.py --scratch ../diream-tests/operators --suite operators
python -I -B tests/fixture_inference_integration.py --output ../diream-tests/crossmodule
```

The core suite uses the CPU requirements. Operator checks also need CPU-capable PyTorch, tqdm, OpenCV and scikit-learn; versions are recorded in `ENVIRONMENT.json`. The cross-module fixture uses NumPy/Pillow and generated examples. Its synthetic draws test interfaces and must not be reported as the scientific bootstrap or additional efficacy samples.

The separate image-resource check and CUDA integration check are described in [INFERENCE.md](INFERENCE.md). Store generated outputs outside the package so the supplied records stay intact. On Windows, use short scratch paths if nested fixture names exceed the filesystem path limit.
