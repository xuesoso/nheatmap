import setuptools

setuptools.setup(
    name='nheatmap',
    version='0.1.0',
    author='YSX',
    author_email='xuesoso@gmail.com',
    packages=setuptools.find_packages(),
    url='https://github.com/xuesoso/nheatmap',
    license='LICENSE',
    description='A painless and customizable tool to generate heatmap with hiearchical clustering results and additional data bars.',
    long_description=open('README.md').read(),
    install_requires=[
       "scipy >= 0.14.0",
       "matplotlib >= 2.0.0",
   ],
)
