"""
This is a boilerplate pipeline 'processing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    homography
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=homography,
            name="homography_func",
            inputs="express_expense",
            outputs="homography_points"
        ),
    ])
