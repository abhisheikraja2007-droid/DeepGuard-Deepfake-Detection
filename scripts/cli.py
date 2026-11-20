"""
CLI Tools for Deepfake Detection Platform
Commands for analysis, training, benchmarking, and ONNX export.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_video(args):
    """Analyze a video file for deepfakes."""
    from inference.engine import InferenceEngine, InferenceConfig
    from models.fusion import DetectionMode
    
    config = InferenceConfig(
        mode=DetectionMode(args.mode),
        device=args.device,
        analyze_audio=not args.no_audio
    )
    
    engine = InferenceEngine(config)
    
    def progress(p, msg):
        print(f"[{p*100:.0f}%] {msg}")
    
    print(f"\nAnalyzing: {args.video}")
    print(f"Mode: {args.mode}, Device: {args.device}\n")
    
    result = engine.analyze_video(args.video, progress_callback=progress)
    
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Fake Probability: {result.overall_fake_probability*100:.1f}%")
    print(f"Risk Level: {result.risk_level.upper()}")
    print(f"Confidence: {result.confidence*100:.1f}%")
    print(f"Video Score: {result.video_score*100:.1f}%")
    print(f"Audio Score: {result.audio_score*100:.1f}%")
    print(f"\nExplanation: {result.explanation}")
    
    if result.suspicious_frames:
        print(f"\nSuspicious Frames: {result.suspicious_frames[:10]}")
    
    return 0


def benchmark(args):
    """Run performance benchmarks."""
    from inference.engine import InferenceEngine, InferenceConfig
    from models.fusion import DetectionMode
    
    print("\nRunning Benchmarks...")
    print("="*50)
    
    modes = ['fast', 'balanced', 'accuracy'] if args.mode == 'all' else [args.mode]
    
    for mode in modes:
        config = InferenceConfig(
            mode=DetectionMode(mode),
            device=args.device
        )
        engine = InferenceEngine(config)
        result = engine.benchmark(num_iterations=args.iterations)
        
        print(f"\n{mode.upper()} MODE:")
        print(f"  FPS: {result['fps']:.1f}")
        print(f"  ms/frame: {result['ms_per_frame']:.2f}")
        print(f"  Device: {result['device']}")
    
    return 0


def export_onnx(args):
    """Export models to ONNX format."""
    import torch
    from models.fast_cnn import FastCNN
    from models.efficient_net import EfficientNetDetector
    from models.xception import XceptionDetector
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = {
        'fast': FastCNN(),
        'balanced': EfficientNetDetector(),
        'accuracy': XceptionDetector()
    }
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    for name, model in models.items():
        if args.model != 'all' and args.model != name:
            continue
        
        model.eval()
        output_path = output_dir / f"{name}_detector.onnx"
        
        print(f"Exporting {name} model to {output_path}...")
        
        torch.onnx.export(
            model, dummy_input, str(output_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            opset_version=12
        )
        
        print(f"  ✓ Exported successfully")
    
    if args.verify:
        print("\nVerifying ONNX models...")
        import onnx
        for onnx_file in output_dir.glob("*.onnx"):
            model = onnx.load(str(onnx_file))
            onnx.checker.check_model(model)
            print(f"  ✓ {onnx_file.name} is valid")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze video file")
    analyze_parser.add_argument("video", help="Path to video file")
    analyze_parser.add_argument("--mode", choices=['fast', 'balanced', 'accuracy'],
                               default='balanced', help="Detection mode")
    analyze_parser.add_argument("--device", default='cpu', help="Device (cpu/cuda)")
    analyze_parser.add_argument("--no-audio", action='store_true', help="Skip audio analysis")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--mode", choices=['fast', 'balanced', 'accuracy', 'all'],
                             default='all', help="Mode to benchmark")
    bench_parser.add_argument("--device", default='cpu', help="Device")
    bench_parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    
    # Export ONNX command
    export_parser = subparsers.add_parser("export-onnx", help="Export to ONNX")
    export_parser.add_argument("--output", default="./models/onnx", help="Output directory")
    export_parser.add_argument("--model", choices=['fast', 'balanced', 'accuracy', 'all'],
                              default='all', help="Model to export")
    export_parser.add_argument("--verify", action='store_true', help="Verify exported models")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        return analyze_video(args)
    elif args.command == "benchmark":
        return benchmark(args)
    elif args.command == "export-onnx":
        return export_onnx(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
