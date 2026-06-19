# Contributing to FlashOCR

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/FlashVision/FlashOCR.git
cd FlashOCR
pip install -e ".[dev,all]"
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run lint: `ruff check flashocr/`
4. Run tests: `flashocr check`
5. Commit and push
6. Open a Pull Request

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting (line length: 120)
- Type hints are encouraged
- Docstrings for all public functions (Google style)
- No hardcoded file paths — use relative or configurable paths

## Adding a New Solution

1. Create `flashocr/solutions/your_solution.py`
2. Follow the existing pattern: accept `predictor` + optional config
3. Implement `process_image(image)` → `(text, confidence)`
4. Implement `get_results()` and `reset()`
5. Add to `flashocr/solutions/__init__.py`

## Adding a New Decoder

1. Create `flashocr/models/decoder/your_decoder.py`
2. Implement `forward(features)` → `(output, lengths)`
3. Implement `decode(output)` → `str`
4. Add to `flashocr/models/decoder/__init__.py`

## Commit Messages

Use clear, descriptive messages:
- `Add attention decoder module`
- `Fix CTC blank token handling in predictor`
- `Update README with LoRA examples`

## Reporting Issues

- Use GitHub Issues
- Include: Python version, PyTorch version, GPU info, error traceback
- Run `flashocr settings` and paste the output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
