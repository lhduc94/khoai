import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="khoai",
    version="0.0.9_1",
    author="Lê Huỳnh Đức",
    author_email="lhduc94@gmail.com",
    description="A small datascience package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lhduc94/khoai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
