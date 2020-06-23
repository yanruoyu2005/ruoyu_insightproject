import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="linearbunny",
    version="0.0.5",
    author="Ruoyu Yan",
    author_email="yanruoyu2005@gmail.com",
    description="A linearbunny package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GPLv3+',
    url="https://github.com/yanruoyu2005/ruoyu_insightproject/linearbunny",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
) 

