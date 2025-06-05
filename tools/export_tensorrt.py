"""Convert a SavedModel to TensorRT for Jetson deployment.

Usage:
  python export_tensorrt.py --model_dir <saved_model> --output_dir <trt_model> [--fp16]
"""
import argparse
import tensorflow as tf


def convert(model_dir: str, output_dir: str, fp16: bool = True) -> None:
    """Convert a TensorFlow SavedModel to TensorRT format."""
    precision = "FP16" if fp16 else "FP32"
    converter = tf.experimental.tensorrt.Converter(
        input_saved_model_dir=model_dir,
        conversion_params=tf.experimental.tensorrt.ConversionParams(precision_mode=precision)
    )
    converter.convert()
    converter.build(input_fn=lambda: [tf.zeros([1, 32, 32, 1], tf.float32)])
    converter.save(output_dir)
    print(f"Saved TensorRT model to {output_dir} (precision={precision})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to SavedModel")
    parser.add_argument("--output_dir", required=True, help="Directory to save TensorRT model")
    parser.add_argument("--fp16", action="store_true", help="Convert using FP16 precision")
    args = parser.parse_args()
    convert(args.model_dir, args.output_dir, args.fp16)


if __name__ == "__main__":
    main()
