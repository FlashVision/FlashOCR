"""FlashOCR CLI — command-line interface for training, validation, prediction, and export."""

import argparse
import sys


def _colored(text, color):
    """Simple ANSI color helper."""
    colors = {"green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m", "red": "\033[91m", "bold": "\033[1m"}
    return f"{colors.get(color, '')}{text}\033[0m"


def _print_banner():
    print(_colored("FlashOCR", "bold") + f" v{_get_version()}")
    print(_colored("Ultra-lightweight text recognition (OCR) framework", "blue"))
    print()


def _get_version():
    from flashocr import __version__
    return __version__


def cmd_version(args):
    """Print version info."""
    _print_banner()


def cmd_settings(args):
    """Print system settings and environment info."""
    import torch
    import platform
    import numpy as np

    _print_banner()
    print(_colored("System", "bold"))
    print(f"  Python:      {platform.python_version()}")
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Machine:     {platform.machine()}")
    print()
    print(_colored("Dependencies", "bold"))
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  CUDA:        {torch.version.cuda or 'Not available'}")
    print(f"  cuDNN:       {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print()
    print(_colored("Hardware", "bold"))
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM:        {mem:.1f} GB")
    else:
        print("  GPU:         None (CPU only)")
    print(f"  CPU cores:   {__import__('os').cpu_count()}")


def cmd_check(args):
    """Verify installation — imports, GPU, and basic inference."""
    _print_banner()
    errors = []

    print(_colored("Checking installation...", "bold"))
    print()

    try:
        import flashocr  # noqa: F401
        print(f"  {_colored('✓', 'green')} flashocr package")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} flashocr package: {e}")
        errors.append(str(e))

    try:
        from flashocr.engine import Trainer, Predictor, Exporter, Validator  # noqa: F401
        print(f"  {_colored('✓', 'green')} engine (Trainer, Predictor, Exporter, Validator)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} engine: {e}")
        errors.append(str(e))

    try:
        from flashocr.solutions import PlateReader, DocumentScanner, ReceiptParser  # noqa: F401
        print(f"  {_colored('✓', 'green')} solutions (PlateReader, DocumentScanner, ReceiptParser)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} solutions: {e}")
        errors.append(str(e))

    try:
        from flashocr.analytics import Benchmark, Profiler  # noqa: F401
        print(f"  {_colored('✓', 'green')} analytics (Benchmark, Profiler)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} analytics: {e}")
        errors.append(str(e))

    try:
        import torch
        from flashocr.cfg import get_config
        from flashocr.models import build_model
        cfg = get_config(model_size="m", input_height=32, input_width=128)
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 128)
            model(dummy)
        print(f"  {_colored('✓', 'green')} model forward pass (FlashOCR-m, 32x128)")
    except Exception as e:
        print(f"  {_colored('✗', 'red')} model forward pass: {e}")
        errors.append(str(e))

    import torch
    if torch.cuda.is_available():
        print(f"  {_colored('✓', 'green')} CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  {_colored('⚠', 'yellow')} No CUDA GPU (training will be slow)")

    print()
    if errors:
        print(_colored(f"✗ {len(errors)} check(s) failed", "red"))
        sys.exit(1)
    else:
        print(_colored("✓ All checks passed! FlashOCR is ready.", "green"))


def cmd_train(args):
    """Train a FlashOCR model."""
    from flashocr.engine.trainer import Trainer

    if args.config:
        from flashocr.cfg import load_yaml_config
        cfg = load_yaml_config(args.config)
        print(f"{_colored('Config:', 'bold')} {args.config}")
        trainer = Trainer(config=cfg, device=args.device)
    else:
        if not args.train_images or not args.train_labels:
            print(_colored("Error:", "red") + " --train-images and --train-labels are required (or use --config)")
            sys.exit(1)
        if not args.val_images or not args.val_labels:
            print(_colored("Error:", "red") + " --val-images and --val-labels are required (or use --config)")
            sys.exit(1)

        kwargs = {
            "model_size": args.model_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "train_images": args.train_images,
            "train_labels": args.train_labels,
            "val_images": args.val_images,
            "val_labels": args.val_labels,
            "save_dir": args.save_dir,
        }
        if args.charset:
            kwargs["charset"] = args.charset
        if args.lora:
            kwargs["lora"] = True
        if args.qlora:
            kwargs["qlora"] = True
        if args.amp:
            kwargs["amp"] = True
        if args.lr:
            kwargs["lr"] = args.lr
        if args.workers is not None:
            kwargs["workers"] = args.workers
        trainer = Trainer(**kwargs)

    trainer.train()


def cmd_predict(args):
    """Run OCR inference on an image or directory."""
    from flashocr.engine.predictor import Predictor

    predictor = Predictor(
        model_path=args.model,
        device=args.device,
    )

    results = predictor.predict(args.source, output_dir=args.output)

    if isinstance(results, tuple) and len(results) == 2:
        text, conf = results
        print(f"\n{_colored('Result:', 'green')} '{text}' (confidence: {conf:.3f})")
    elif isinstance(results, list):
        print(f"\n{_colored(f'Recognized {len(results)} images:', 'green')}")
        for r in results:
            print(f"  {r['file']}: '{r['text']}' ({r['confidence']:.3f})")


def cmd_val(args):
    """Validate model on a dataset."""
    from flashocr.engine.validator import Validator
    validator = Validator(
        model_path=args.model,
        val_images=args.val_images,
        val_labels=args.val_labels,
        device=args.device,
    )
    results = validator.validate()
    print(f"\n{_colored('Results:', 'green')}")
    print(f"  CER:      {results['cer']:.4f}")
    print(f"  WER:      {results['wer']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Loss:     {results['val_loss']:.4f}")


def cmd_export(args):
    """Export model to ONNX."""
    from flashocr.engine.exporter import Exporter
    exporter = Exporter(model_path=args.model)
    path = exporter.export(output=args.output, simplify=args.simplify)
    print(f"\n{_colored('✓', 'green')} Exported: {path}")


def main():
    parser = argparse.ArgumentParser(
        prog="flashocr",
        description="FlashOCR: Ultra-lightweight text recognition (OCR) framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashocr check                                       Verify installation
  flashocr train --train-images data/train --train-labels data/train/labels.tsv \\
                 --val-images data/val --val-labels data/val/labels.tsv
  flashocr predict --model best.pth --source word.jpg
  flashocr export --model best.pth --output model.onnx --simplify

Documentation: https://github.com/FlashVision/FlashOCR
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show version info")

    # settings
    subparsers.add_parser("settings", help="Show system settings (Python, PyTorch, CUDA, GPU)")

    # check
    subparsers.add_parser("check", help="Verify installation and run health check")

    # train
    train_p = subparsers.add_parser("train", help="Train a FlashOCR model")
    train_p.add_argument("--config", default=None, help="Path to YAML config file")
    train_p.add_argument("--model-size", default="m", choices=["m-0.5x", "m", "m-1.5x"],
                         help="Model variant (default: m)")
    train_p.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    train_p.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    train_p.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_p.add_argument("--charset", default=None,
                         help="Character set string (default: 0-9a-z)")
    train_p.add_argument("--device", default="cuda", help="Device: cuda or cpu (default: cuda)")
    train_p.add_argument("--train-images", default=None, help="Path to training images directory")
    train_p.add_argument("--train-labels", default=None, help="Path to training labels TSV file")
    train_p.add_argument("--val-images", default=None, help="Path to validation images directory")
    train_p.add_argument("--val-labels", default=None, help="Path to validation labels TSV file")
    train_p.add_argument("--save-dir", default="workspace/ocr_train", help="Output directory")
    train_p.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    train_p.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    train_p.add_argument("--qlora", action="store_true", help="Enable QLoRA fine-tuning")
    train_p.add_argument("--amp", action="store_true", help="Enable mixed precision (FP16)")

    # predict
    pred_p = subparsers.add_parser("predict", help="Run OCR inference on image or directory")
    pred_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    pred_p.add_argument("--source", required=True, help="Image path or directory")
    pred_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    pred_p.add_argument("--output", default=None, help="Output directory for results")

    # val
    val_p = subparsers.add_parser("val", help="Validate model on dataset")
    val_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    val_p.add_argument("--val-images", required=True, help="Path to validation images")
    val_p.add_argument("--val-labels", required=True, help="Path to validation labels TSV")
    val_p.add_argument("--device", default="cuda", help="Device (default: cuda)")

    # export
    exp_p = subparsers.add_parser("export", help="Export model to ONNX format")
    exp_p.add_argument("--model", required=True, help="Path to .pth checkpoint")
    exp_p.add_argument("--output", default="model.onnx", help="Output path (default: model.onnx)")
    exp_p.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")

    args = parser.parse_args()

    if args.command is None:
        _print_banner()
        parser.print_help()
        sys.exit(0)

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "train": cmd_train,
        "predict": cmd_predict,
        "val": cmd_val,
        "export": cmd_export,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
