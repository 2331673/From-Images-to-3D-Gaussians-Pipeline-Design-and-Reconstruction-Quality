"""Export reproducibility records without personal filesystem locations."""
from pathlib import Path
import re


def sanitize_text(text, roots=()):
    """Replace explicitly mapped roots first; redact other mounted/home paths."""
    mappings = sorted(((str(Path(src).resolve()), dest) for src, dest in roots),
                      key=lambda pair: len(pair[0]), reverse=True)
    for source, target in mappings:
        text = text.replace(source + '/', target.rstrip('/') + '/' if target else '')
        text = text.replace(source, target or '.')
    return re.sub(r'''/(?:mnt|home|root)/[^\s"'<>`,;)\]}]+''',
                  '[LOCAL_PATH_REDACTED]', text)


def sanitize_record(path, roots=()):
    path = Path(path)
    before = path.read_text()
    after = sanitize_text(before, roots)
    if after != before:
        path.write_text(after)


def sanitize_results(repository, experiment):
    repository, experiment = Path(repository).resolve(), Path(experiment).resolve()
    roots = [(experiment, 'experiments/' + experiment.name), (repository, '')]
    for path in (repository / 'results').rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.yml', '.yaml', '.txt', '.csv'}:
            sanitize_record(path, roots)
