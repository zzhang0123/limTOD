# limTOD documentation

The rendered documentation lives at
**<https://limtod.readthedocs.io>**.

The pages in this directory are the Sphinx (MyST-Markdown) sources —
build locally with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Release notes live in
[CHANGELOG.md](https://github.com/zzhang0123/limTOD/blob/main/CHANGELOG.md);
coordinate and beam-orientation conventions in
[theory.md](theory.md).
