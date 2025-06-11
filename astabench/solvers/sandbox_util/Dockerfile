FROM python:3.11-bookworm

# ds1000 is a pretty hefty install and should never change since it's
# a third-party eval, so let's do that first
COPY ds1000_requirements.txt .
RUN pip install --no-cache-dir -r ds1000_requirements.txt

COPY super_requirements.txt .
RUN pip install --no-cache-dir -r super_requirements.txt

RUN pip install inspect-tool-support && inspect-tool-support post-install
RUN pip install "mcp~=1.9.1"

RUN pip install jupyter_kernel_gateway requests numpy pandas
RUN pip install jupyter_client notebook

# Datavoyager requirement
RUN pip install "matplotlib==3.10.0"
