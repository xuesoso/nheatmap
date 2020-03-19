import setuptools

setuptools.setup(
    name='nheatmap',
    version='0.1.2',
    author='YSX',
    author_email='xuesoso@gmail.com',
    packages=setuptools.find_packages(),
    url='https://github.com/xuesoso/nheatmap',
    license='LICENSE',
    description='A painless and customizable tool to generate heatmap with hiearchical clustering results and additional data bars.',
    long_description=open('README.md').read(),
    keywords = ['heatmap', 'hiearchical clustering', 'neat heatmap'],
    install_requires=[
       "matplotlib >= 3.0.3",
       "scipy >= 1.3.1",
       "numpy >= 1.17.2",
       "pandas >= 0.25.1",
   ],
)
