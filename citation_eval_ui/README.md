# Citation Evaluation UI

A web interface for reviewing and labeling LLM-based citation evaluation assessments.

## Installation

```bash
cd citation_eval_ui
uv sync
```

## Usage

### Option 1: Using the launch script (recommended)
```bash
python run.py
```

### Option 2: Using uv with environment variable
```bash
PYTHONPATH=/data/new_astabench/citation_eval_ui uv run python -c "from citation_eval_ui.main import main; main()"
```

Then open http://localhost:8000 in your browser.

## Features

- **File Selection**: Load citation evaluation CSV files from `test_dvc_logs/debug_logs/`
- **Question Review**: View the original questions being answered
- **Text Display**: See the generated text with claims highlighted
- **Citation Assessment**: Review supporting and non-supporting citations for each claim
- **Interactive Annotation**: 
  - Label citations as supporting/non-supporting/irrelevant
  - Add notes about missing citations
  - Navigate between claims and files
- **Progress Tracking**: Visual progress indicators and keyboard shortcuts
- **Auto-save**: Annotations are automatically saved to JSON files
- **Responsive Design**: Works on desktop and mobile devices

## Data Format

The application expects CSV files with columns:
- `question`: The question being answered
- `text`: The generated section text
- `citations`: List of citation objects with `id` and `snippets`
- `claims`: List of claim objects with `text`, `supporting`, `non_supporting`, and `is_fully_supported`

## Keyboard Shortcuts

- `←/→`: Navigate between claims
- `↑/↓`: Navigate between rows
- `Ctrl+S`: Save annotations
- `?`: Show help modal

## Output

Annotations are saved as JSON files in the `annotations/` directory, with one file per CSV file being annotated.