from setuptools import setup, find_packages

setup(
    name="build_a_rest_api_with_user_authentication_and_rate_limiting",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "app=main:main",
        ],
    },
)