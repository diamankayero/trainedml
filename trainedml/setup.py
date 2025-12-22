from setuptools import setup, find_packages

setup(
    name="trainedml",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "requests",
        "plotly; extra == 'plotly'"
    ],
    entry_points={
        "console_scripts": [
            "trainedml=trainedml:main"
        ]
    },
    author="diamankayero",
    description="Outils pour charger, entraîner, comparer et visualiser des modèles ML.",
    license="MIT",
)
