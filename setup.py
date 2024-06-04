from setuptools import setup, find_packages

__version__ = '0.0.1'
install_requires = []

setup(
    name='latent_matching',
    version=__version__,
    description='Latent Matching',

    url='https://github.com/grustanford/latent_matching',

    # Author details
    author='Hyunwoo Gu',
    author_email='hwgu@stanford.edu',

    packages=find_packages(where='src/smm'),
    install_requires=install_requires
)