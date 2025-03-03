import io
import os
import setuptools

here = os.path.realpath(os.path.dirname(__file__))

name = 'polysigkernel'

version = '0.0.1'

author = 'Francesco Piatti'

author_email = 'piatti.francesco@outlook.com'

description = "Polynomial-based Schemes for Signature Kernels"

with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

url = "https://github.com/FrancescoPiatti/polysigkernel"

license = "Apache-2.0"

classifiers = ["Intended Audience :: Developers",
               "Intended Audience :: Financial and Insurance Industry",
               "Intended Audience :: Information Technology",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: Apache Software License",
               "Natural Language :: English",
               "Operating System :: MacOS :: MacOS X",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Programming Language :: Python :: 3",
               "Topic :: Scientific/Engineering :: Artificial Intelligence",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]

python_requires = "~=3.12"

install_requires = ["jax >= 0.4.23"]


setuptools.setup(name=name,
                 version=version,
                 author=author,
                 author_email=author_email,
                 maintainer=author,
                 maintainer_email=author_email,
                 description=description,
                 long_description=readme,
                 url=url,
                 license=license,
                 classifiers=classifiers,
                 zip_safe=False,
                 python_requires=python_requires,
                 install_requires=install_requires,
                 packages=[name])
