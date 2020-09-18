from setuptools import setup

with open("README.md","r") as fh:
  ld = fh.read()

setup(
  name = 'scikit-XCS',
  packages = ['skXCS'],
  version = '1.0.6',
  license='License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
  description = 'XCS Learning Classifier System',
  long_description_content_type="text/markdown",
  author = 'Robert Zhang, Ryan J. Urbanowicz',
  author_email = 'robertzh@seas.upenn.edu,ryanurb@upenn.edu',
  url = 'https://github.com/UrbsLab/scikit-XCS',
  download_url = 'https://github.com/UrbsLab/scikit-XCS/archive/v_1.0.6.tar.gz',
  keywords = ['machine learning','data analysis','data science','learning classifier systems','xcs'],
  install_requires=['numpy','pandas','scikit-learn'],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'Topic :: Utilities',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
  long_description=ld
)