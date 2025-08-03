"""
Setup script for Al-artworks.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="al-artworks",
    version="1.0.0",
    author="Al-artworks Team",
    author_email="team@al-artworks.com",
    description="Qt6-based image-editing app with Eve, the celestial creative goddess",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/al-artworks/al-artworks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Graphics :: Editors",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Environment :: X11 Applications :: Qt",
    ],
    python_requires=">=3.13",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "al-artworks=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "al-artworks": [
            "assets/**/*",
            "data/**/*",
            "config/*.yaml",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/al-artworks/al-artworks/issues",
        "Source": "https://github.com/al-artworks/al-artworks",
        "Documentation": "https://al-artworks.readthedocs.io",
    },
)