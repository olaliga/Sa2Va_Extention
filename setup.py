from setuptools import setup, find_packages

setup(
    name="sa2va-models",
    version="0.1.0",
    description="Sa2VA Models Extension - A modular framework for visual language models and segmentation",
    author="Shi Jie Huang",
    author_email="iphone11134@gmail.com",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pillow>=9.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 