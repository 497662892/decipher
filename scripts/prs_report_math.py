"""Build-time KaTeX rendering with embedded fonts for standalone offline reports."""
from __future__ import annotations

import base64
import json
import re
import subprocess
from pathlib import Path

from decipehr.external_pgs import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/report_vendor/katex_0_19_1/node_modules/katex"


def render_latex(expressions):
    """Return CSS, accessible HTML formulas and renderer provenance; reject invalid TeX."""
    if not PACKAGE.is_dir():
        raise FileNotFoundError(
            "Install the pinned renderer: npm install --prefix outputs/report_vendor/katex_0_19_1 "
            "--cache outputs/npm-cache --ignore-scripts --no-audit --no-fund --save-exact katex@0.19.1"
        )
    version = json.loads((PACKAGE / "package.json").read_text())["version"]
    assert version == "0.19.1"
    tex = [e.strip().removeprefix("$").removesuffix("$") for e in expressions]
    script = r"""
const fs=require('fs'),katex=require(process.argv[1]);
const expressions=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(expressions.map(tex=>katex.renderToString(tex,{
 displayMode:true,output:'htmlAndMathml',throwOnError:true,strict:'error',trust:false
}))));
"""
    process = subprocess.run(["node", "-e", script, str(PACKAGE / "dist/katex.js")],
                             input=json.dumps(tex), text=True, capture_output=True, check=True)
    formulas = ['<div class="formula">'+markup+'</div>' for markup in json.loads(process.stdout)]
    inputs = [Path(__file__), PACKAGE / "package.json", PACKAGE / "dist/katex.js",
              PACKAGE / "dist/katex.min.css", PACKAGE / "LICENSE"]

    def embed_font(match):
        body = match[1]
        filename = re.search(r"url\(fonts/([^()]+\.woff2)\)", body)[1]
        path = PACKAGE / "dist/fonts" / filename
        inputs.append(path)
        encoded = base64.b64encode(path.read_bytes()).decode()
        return "@font-face{"+re.sub(r"src:[^;]+", 'src:url(data:font/woff2;base64,'+encoded+') format("woff2")', body)+"}"

    css = re.sub(r"@font-face\{([^}]+)\}", embed_font, (PACKAGE / "dist/katex.min.css").read_text())
    assert "url(fonts/" not in css
    css += """
.formula{overflow-x:auto;padding:18px 20px;margin:16px 0;background:#f8fafc;border:1px solid #e3eaf0;border-radius:8px}
.formula .katex-display{margin:0}.formula .katex-display>.katex{text-align:left}
.formula .katex{font-size:1.12em}.formula .katex-mathml{user-select:none}
@media(max-width:650px){.formula{padding:14px 12px}.formula .katex{font-size:1.02em}}
"""
    license_text = (PACKAGE / "LICENSE").read_text()
    style = '<!-- KaTeX '+version+'\n'+license_text+' --><style data-math-renderer="katex">'+css+'</style>'
    metadata = {"engine": "KaTeX", "version": version, "output": "htmlAndMathml",
                "render_time": "build", "network_required_to_view": False, "inline_woff2_fonts": len(inputs)-5,
                "formulas": len(tex), "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs}}
    return style, formulas, metadata


def check_latex(page, count):
    assert page.count('class="katex"') == count
    assert page.count('<annotation encoding="application/x-tex">') == count
    assert page.count('class="katex-mathml"') == count
    assert 'class="katex-error"' not in page
    assert 'data-math-renderer="katex"' in page
    assert "url(fonts/" not in page
