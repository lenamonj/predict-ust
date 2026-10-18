from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK = Path(__file__).parents[1] / "notebook" / "UST_RV_Score.ipynb"


@pytest.mark.network
def test_notebook_runs_clean_top_to_bottom():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ExecutePreprocessor(timeout=900, kernel_name="ust_rv").preprocess(nb, {"metadata": {"path": str(NOTEBOOK.parent)}})
    outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])]
    assert not [o for o in outputs if o.output_type == "error"]
    assert not [o for o in outputs if o.output_type == "stream" and o.name == "stderr"]
    html = "".join(str(o.get("data", {}).get("text/html", "")) for o in outputs)
    assert "RV 2y" in html and "RV 10y" in html
    # every chart title carries an as-of date
    figs = [o["data"]["application/vnd.plotly.v1+json"] for o in outputs if "application/vnd.plotly.v1+json" in o.get("data", {})]
    assert len(figs) == 29
    assert all("as of 20" in f["layout"]["title"]["text"] for f in figs)
