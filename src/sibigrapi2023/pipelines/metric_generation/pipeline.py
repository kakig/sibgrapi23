"""
This is a boilerplate pipeline 'metric_generation'
generated using Kedro 0.18.7
"""

from .nodes import (
    generate_method_metrics,
    generate_ablation_metrics,
)

from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])
