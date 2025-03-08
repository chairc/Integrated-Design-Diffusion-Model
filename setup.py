from setuptools import setup, find_packages


def get_install_requirements():
    """
    Get package from requirements.txt
    """
    with open(file="requirements.txt", mode="r", encoding="utf-8") as f:
        packages = [x.strip() for x in f.read().splitlines()]
    # Remove comments
    packages = [x for x in packages if not x.startswith("#")]
    return packages


def get_long_description():
    with open(file="README.md", mode="r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description


if __name__ == "__main__":
    package_list = [
        "coloredlogs==15.0.1",
        "gradio==5.0.0",
        "matplotlib==3.7.1",
        "numpy==1.25.0",
        "Pillow==10.3.0",
        "Requests==2.32.0",
        "scikit-image==0.22.0",
        "torch_summary==1.4.5",
        "tqdm==4.66.3",
        "pytorch_fid==0.3.0",
        "fastapi==0.115.6",
        "tensorboardX==2.6.1",
        "torch>=1.9.0",
        "torchvision>=0.10.0"
    ]
    # Define the metadata of the package
    setup(
        name="iddm",
        version="1.1.8-b3",
        packages=find_packages(),
        python_requires=">=3.8",
        install_requires=package_list,
        license="Apache-2.0",
        description="IDDM: Integrated Design Diffusion Model",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="chairc",
        author_email="chenyu1998424@gmail.com",
        url="https://github.com/chairc/Integrated-Design-Diffusion-Model",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        project_urls={
            "Documentation": "https://github.com/chairc/Integrated-Design-Diffusion-Model/blob/main/README.md",
            "Source": "https://github.com/chairc/Integrated-Design-Diffusion-Model",
            "Tracker": "https://github.com/chairc/Integrated-Design-Diffusion-Model/issues",
        },
    )
