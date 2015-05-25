from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='ToSidewalk',
    version=0.1,
    url='',
    license='Apache Software License',
    author='Zachary Lawrence and Kotaro Hara',
    tests_require=['pytest'],
    install_requires=['numpy>=1.9.2'],
    cmdclass={},
    author_email='koe.bluebear@gmail.com',
    description='The program generates potential sidewalk network from OpenStreetMap street data',
    long_description='',
    packages=['ToSidewalk'],
    include_package_data=True,
    platforms='any',
    test_suite='',
    classifiers=[],
    extra_require={}
)
