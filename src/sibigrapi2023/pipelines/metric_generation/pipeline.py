"""
This is a boilerplate pipeline 'metric_generation'
generated using Kedro 0.18.7
"""

from .nodes import (
    generate_methods_ocr_text,
    generate_ablation_ocr_text,
    generate_pipeline_sample_images,
)

from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_methods_ocr_text,
            name="generate_methods_ocr_original_text_func",
            inputs="express_test",
            outputs="methods_original_ocr_text"
        ),
        node(
            func=generate_methods_ocr_text,
            name="generate_methods_ocr_distorted_text_func",
            inputs="distorted",
            outputs="methods_distorted_ocr_text"
        ),
        node(
            func=generate_ablation_ocr_text,
            name="generate_ablation_ocr_text_func",
            inputs="express_test",
            outputs="ablation_ocr_text"
        ),
        node(
            func=generate_ablation_ocr_text,
            name="generate_ablation_ocr_distorted_text_func",
            inputs="distorted",
            outputs="ablation_ocr_distorted_text",
        ),
        node(
            func=generate_pipeline_sample_images,
            name="generate_pipeline_sample_images_func",
            inputs="express_expense",
            outputs="pipeline_sample_images",
        ),
    ])
