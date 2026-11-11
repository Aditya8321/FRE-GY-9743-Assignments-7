"""Execute every cell in a fresh IPython process without opening kernel sockets.

Run from the repository root: python scripts/execute_notebook.py
This is useful in restricted environments; ordinary Jupyter Run All also works.
The notebook is saved only after every code cell has succeeded.
"""
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'hw7-mpl'))
os.environ.setdefault('IPYTHONDIR', str(Path(tempfile.gettempdir()) / 'hw7-ipython'))
# System-site-packages venvs may not include the base interpreter's Jupyter data.
base_data = Path(sys.base_prefix) / 'share' / 'jupyter'
if base_data.is_dir():
    os.environ['JUPYTER_PATH'] = os.pathsep.join(
        [str(base_data)] + ([os.environ['JUPYTER_PATH']] if os.environ.get('JUPYTER_PATH') else [])
    )

import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support


def main():
    path = ROOT / 'hw7_monotone_convex_interpolator.ipynb'
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    configure_inline_support(shell, 'inline')
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            print(captured.stdout, file=sys.stderr)
            print(captured.stderr, file=sys.stderr)
            raise RuntimeError(f'Notebook failed in code cell {count}') from error
        cell.execution_count = count
        cell.outputs = [nbformat.v4.new_output('display_data', data=o.data, metadata=o.metadata)
                        for o in captured.outputs]
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
    notebook.metadata.language_info = {'name': 'python', 'version': sys.version.split()[0]}
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f'Successfully executed {count} code cells in a fresh IPython process.')
    from nbconvert import HTMLExporter
    html, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
    output = ROOT / 'submission' / 'Assignment_7_as22008.html'
    output.write_text(html, encoding='utf-8')
    print(f'Exported {output.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
