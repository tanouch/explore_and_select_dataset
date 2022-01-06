from setuptools import setup, find_packages

setup(
  name = 'explore_your_dataset',
  packages = find_packages(),
  include_package_data = True,
  version = '0.14.3',
  license='MIT',
  description = 'Explore and Select your dataset',
  authors = 'Ugo Tanielian',
  url = 'https://github.com/lucidrains/dalle-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'text-to-image'
  ],
  install_requires=[
    'numpy',
    'pyarrow',
    'Pillow',
    'fire',
    'img2dataset',
    'scipy',
    'pandas',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'tqdm',
    'youtokentome',
    'WebDataset', 
    'matplotlib', 
    'ipywidgets', 
    'clip-anytorch', 
    'dalle-pytorch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
