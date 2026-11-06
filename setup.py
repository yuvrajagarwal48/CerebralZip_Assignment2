from setuptools import setup, find_packages

setup(
    name="forensic_license_plate_pipeline",
    version="2.0.0",
    description="Autonomous Forensic License Plate Super-Resolution, Restoration & OCR Pipeline",
    author="Antigravity DeepMind Team",
    packages=find_packages(),
    py_modules=["forensic_pipeline", "run_pipeline", "app"],
    install_requires=[
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "rapidocr-onnxruntime>=1.2.0",
        "scikit-image>=0.21.0",
        "huggingface-hub>=0.16.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "onnxruntime",
        "streamlit>=1.24.0"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "forensic-anpr=run_pipeline:main",
        ],
    },
)

