from setuptools import setup, find_packages

setup(
    name='msc',
    version='1.0.0',
    url='https://github.com/noamsgl/msc',
    author='Noam Siegel',
    author_email='noamsi@post.bgu.ac.il',
    description='school stuff',
    packages=find_packages(),    
    install_requires=['numpy >= 1.16.1', 'matplotlib >= 3.3.4', 'mne >= 0.23.4'],
)
