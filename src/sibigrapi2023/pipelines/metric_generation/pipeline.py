"""
This is a boilerplate pipeline 'metric_generation'
generated using Kedro 0.18.7
"""

from .nodes import (
    generate_methods_ocr_text,
    generate_ablation_metrics,
)

from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_methods_ocr_text,
            name="generate_methods_ocr_text_func",
            inputs="express_small",
            outputs="methods_ocr"
        ),
    ])
