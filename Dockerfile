FROM condaforge/mambaforge:24.3.0-0

RUN mamba create -y -n mod python=3.13 && echo "mamba activate mod" >> /root/.bashrc
SHELL ["bash", "-lc"]

WORKDIR /app
COPY environment.yml /app/

RUN mamba env update -n mod -f environment.yml

COPY . /app
EXPOSE 8501
CMD ["bash", "-lc", "mamba activate app && streamlit run inference.py --server.port=8501 --server.address=0.0.0.0"]