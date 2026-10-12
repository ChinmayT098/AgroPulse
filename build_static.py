"""
Static Site Export for Netlify
------------------------------
Netlify serves static files and cannot run a long-lived Flask server, so this
script pre-renders every Flask page to HTML using the Flask test client and
exports the trained Random Forest model to JSON. The prediction page then runs
the exact same forest in the browser (see static/js/predict.js).

Usage:
    python build_static.py        # writes the site to ./dist
"""

import os
import json
import shutil

from app import app
from model import get_model_manager, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, 'dist')

PAGES = {
    '/': 'index.html',
    '/analysis': 'analysis/index.html',
    '/predict': 'predict/index.html',
    '/spark': 'spark/index.html',
}


def export_model(path):
    """Serialize the fitted pipeline (one-hot encoder + forest) to compact JSON."""
    manager = get_model_manager()
    pipeline = manager.pipeline
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    rf = pipeline.named_steps['regressor']

    trees = []
    for est in rf.estimators_:
        t = est.tree_
        trees.append({
            'l': t.children_left.tolist(),
            'r': t.children_right.tolist(),
            'f': t.feature.tolist(),
            # Thresholds are stored at full precision to match sklearn's float32 comparisons
            't': [float(x) for x in t.threshold],
            'v': [round(float(x), 5) for x in t.value[:, 0, 0]],
        })

    payload = {
        'categorical_features': CATEGORICAL_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
        'categories': {
            col: [str(c) for c in cats]
            for col, cats in zip(CATEGORICAL_FEATURES, ohe.categories_)
        },
        'crop_stats': manager.crop_stats,
        'metrics': manager.get_metrics(),
        'trees': trees,
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'))


def build():
    if os.path.exists(DIST_DIR):
        shutil.rmtree(DIST_DIR)
    os.makedirs(DIST_DIR)

    app.jinja_env.globals['static_export'] = True
    client = app.test_client()

    for route, out_file in PAGES.items():
        response = client.get(route)
        if response.status_code != 200:
            raise RuntimeError(f"Rendering {route} failed with status {response.status_code}")
        out_path = os.path.join(DIST_DIR, out_file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(response.data)
        print(f"Rendered {route} -> dist/{out_file}")

    shutil.copytree(os.path.join(BASE_DIR, 'static'), os.path.join(DIST_DIR, 'static'))

    model_dir = os.path.join(DIST_DIR, 'static', 'model')
    os.makedirs(model_dir, exist_ok=True)
    export_model(os.path.join(model_dir, 'forest.json'))
    print("Exported Random Forest model -> dist/static/model/forest.json")


if __name__ == '__main__':
    build()
