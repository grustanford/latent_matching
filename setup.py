from setuptools import setup, find_packages

__version__ = '0.0.1'
install_requires = []

setup(
    name='smm',
    version=__version__,
    description='Semantic Matching Model',

    url='https://github.com/grustanford/semantic_matching',

    # Author details
    author='Hyunwoo Gu',
    author_email='hwgu@stanford.edu',

    packages=find_packages(where='src/smm'),
    install_requires=install_requires
)