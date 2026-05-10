FROM condaforge/miniforge3:latest

LABEL org.opencontainers.image.title="Coregix"
LABEL org.opencontainers.image.description="Pairwise raster coregistration for geospatial imagery"
LABEL org.opencontainers.image.source="https://github.com/iosefa/coregix"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /opt/coregix

COPY pyproject.toml README.md LICENSE ./
COPY coregix ./coregix

RUN conda install -y -c conda-forge \
        "python=3.12" \
        "gdal=3.12.*" \
        pip \
        "setuptools>=61" \
        wheel \
        "affine>=2.4" \
        numpy \
        "rasterio>=1.4.3" \
    && pip install --no-cache-dir . \
    && conda clean -afy

ENTRYPOINT ["vhr-align-image-pair"]
CMD ["--help"]
