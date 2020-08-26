import setuptools
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

version = get_version('./nheatmap/__init__.py')

setuptools.setup(
    name='nheatmap',
    version=version,
    author='YSX',
    author_email='xuesoso@gmail.com',
    packages=setuptools.find_packages(),
    url='https://github.com/xuesoso/nheatmap',
    license='LICENSE',
    description='A painless and customizable tool to generate heatmap with hiearchical clustering results and additional data bars.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords = ['heatmap', 'hiearchical clustering', 'neat heatmap'],
    install_requires=[
       "matplotlib >= 3.0.3",
       "scipy >= 1.3.1",
       "numpy >= 1.17.2",
       "pandas >= 0.25.1",
       "packaging",
   ],
)
